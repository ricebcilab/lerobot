"""A loaded policy, a LIBERO scene, the operator's input and the live view, in one object.

Both entry points drive a `Session`: the REPL changes its scene and mode
between rollouts, the experiment driver walks a schedule of scenes. `rollout`
is the step loop of `lerobot_eval.rollout()` with the user's prompt in place
of the env's task description, an optional action hook (teleop modes) and an
optional per-step recorder (experiments).
"""

import hashlib
import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import torch
from config import MODES, ControlSettings, SessionSettings, gain_that_cannot_clip, matrix_list, spec_label
from live_view import LiveView
from reproducibility import TrialSpec
from teleop import KeyboardReader, PolicyOperator, TeleopChain

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import (
    check_env_attributes_and_types,
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.steering import (
    DEADBAND,
    GRIPPER_DIM,
    N_ACTION_DIMS,
    AdapterModules,
    FlowControlSteering,
    FlowReversalSteering,
    SteeringConfig,
    SteeringPolicy,
)
from lerobot.utils.constants import ACTION

VIDEO_FPS = 30
RATE_HZ = 20  # LIBERO control rate; caps the loop so teleop feels real-time
# Suite defaults (~14 s) are too short for manual driving. Must stay below
# robosuite's horizon of 1000 (which includes the 10 settle steps at reset):
# LIBERO's step() replaces the horizon `done` with task success, so exceeding
# the horizon raises "executing action in terminated episode" instead of ending.
TELEOP_MAX_STEPS = 900
# The shared modes are lerobot's steering policy's (`SteeringPolicy`, the cursor example's too) over pi0.5
STEERING_MODES = {"shared_flow_control": "flow_control", "shared_flow_reversal_steering": "frs"}


@dataclass
class RolloutResult:
    success: bool
    steps: int
    frames: list[np.ndarray] = field(repr=False)
    metrics: dict


class _ZeroPolicy:
    """Stand-in policy for teleop mode: outputs zeros, the teleop hook fills them in."""

    def reset(self) -> None:
        pass

    def select_action(self, obs) -> torch.Tensor:
        return torch.zeros(1, N_ACTION_DIMS)


def _identity(x):
    return x


def _frame(observation) -> np.ndarray:
    """The agentview image of a (vectorised) observation, oriented like LiberoEnv.render().

    Read from the observation rather than render(): the sub-env auto-resets on
    termination, so after a final step render() would show the next fresh scene.
    """
    return observation["pixels"]["image"][0][::-1, ::-1]


def _teleop_hook(source, full: bool) -> Callable[[np.ndarray], np.ndarray]:
    """Paste the teleop command into the env action: its translation (shared override), or the whole of it,
    translation, rotation and gripper (`full`, teleop mode: nothing of the action is the policy's)."""

    def hook(action: np.ndarray) -> np.ndarray:
        action = action.copy()
        action[0, :3] = source.translation
        if full:
            action[0, 3:6] = source.rotation
            action[0, GRIPPER_DIM] = source.gripper
        return action

    return hook


class Session:
    def __init__(self, settings: SessionSettings):
        self.settings = settings
        control = settings.control
        self.tau, self.lam, self.ic_gain = control.tau, control.lam, control.ic_gain  # FC knobs
        self.zeroed_dims = tuple(control.zeroed_dims)

        self.keyboard = KeyboardReader()
        self.chain = TeleopChain(self.keyboard, translation_noise=control.translation_noise)
        # A synthetic operator (the task-prompted policy) replaces the SpaceMouse/keyboard.
        self.operator: PolicyOperator | None = None
        if settings.operator == "policy":
            self.operator = PolicyOperator(settings.n_action_steps, settings.policy_operator)
            self.chain.attach_operator(self.operator)
        self.adapter = AdapterModules(n_dims=None)  # the 7-D holder: each module on its own dims
        self.view = LiveView(settings.port, self.keyboard, self._status_extra)
        self.view.start()
        print(f"\nLive view: {self.view.url}  (VSCode should auto-forward the port)\n")

        logging.info(f"Loading policy {settings.policy_path} ...")
        self.policy_cfg = PreTrainedConfig.from_pretrained(settings.policy_path)
        self.policy_cfg.pretrained_path = settings.policy_path
        self.policy_cfg.n_action_steps = settings.n_action_steps
        if hasattr(self.policy_cfg, "compile_model"):
            self.policy_cfg.compile_model = settings.compile

        self.env_cfg, self.envs_dict, self.vec_env = self._build_env(settings.suite, settings.task_id)
        self.suite, self.task_id = settings.suite, settings.task_id

        self.policy = make_policy(cfg=self.policy_cfg, env_cfg=self.env_cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy_cfg,
            pretrained_path=str(self.policy_cfg.pretrained_path),
            preprocessor_overrides={
                "device_processor": {"device": str(self.policy.config.device)},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=self.env_cfg, policy_cfg=self.policy_cfg
        )
        self.autocast_ctx = (
            torch.autocast(device_type=torch.device(self.policy.config.device).type)
            if self.policy_cfg.use_amp
            else nullcontext()
        )
        if self.operator is not None:
            self.operator_mean, self.operator_std = FlowControlSteering.action_mean_std(self.postprocessor)

        self._zero_policy = _ZeroPolicy()
        self._steering: SteeringPolicy | None = None  # the shared mode's steering policy, built on first use
        self._steering_fields: dict | None = None  # the config fields it was built with
        self.mode = "policy"
        self._scene_names: dict | None = None
        self.apply_control(control)
        self.view.open_in_browser()  # after the model load, so a tab left open by the previous run is noticed

    # ------------------------------------------------------------ scene

    @staticmethod
    @cache
    def list_tasks(suite: str) -> list[str]:
        from lerobot.envs.libero import _get_suite

        return [t.language for t in _get_suite(suite).tasks]

    @staticmethod
    @cache
    def initial_state_count(suite: str, task_id: int) -> int:
        """Read reset counts without constructing an environment or loading the policy."""
        from lerobot.envs.libero import _get_suite, get_task_init_states

        return len(get_task_init_states(_get_suite(suite), task_id))

    @staticmethod
    def _build_env(suite: str, task_id: int):
        """Build a 1-env SyncVectorEnv for one LIBERO scene, mirroring the README's lerobot-eval settings."""
        env_cfg = LiberoEnvConfig(task=suite, task_ids=[task_id])
        envs_dict = make_env(env_cfg, n_envs=1, use_async_envs=False)
        vec_env = envs_dict[suite][task_id]
        check_env_attributes_and_types(vec_env)
        return env_cfg, envs_dict, vec_env

    @property
    def task_description(self) -> str:
        return self.vec_env.envs[0].task_description

    def set_scene(self, suite: str, task_id: int) -> None:
        """Switch scene; raises ValueError / FileNotFoundError and keeps the old scene on failure."""
        env_cfg, envs_dict, vec_env = self._build_env(suite, task_id)
        close_envs(self.envs_dict)
        self.env_cfg, self.envs_dict, self.vec_env = env_cfg, envs_dict, vec_env
        self.suite, self.task_id = suite, task_id
        self._scene_names = None
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=self.policy_cfg
        )

    # ------------------------------------------------------------ scene state (mode recording)

    def _sim(self):
        env = self.vec_env.envs[0]
        env = getattr(env, "unwrapped", env)
        return env._env.env  # lerobot LiberoEnv -> hf-libero OffScreenRenderEnv -> BDDL domain env

    def scene_state(self) -> dict | None:
        """Object positions and articulation joint values, for recording which mode a trial expressed.

        Call it before stepping the env: after a terminating step the vector env has already reset.

        Returns {"object_names", "object_pos" (K, 3), "articulation_names", "articulation_qpos" (J,)}
        or None when the simulator is not reachable. Robot joints are excluded; free joints
        are reported through their body position, slide/hinge joints (drawers, knobs) by value.
        """
        try:
            domain = self._sim()
            sim, model = domain.sim, domain.sim.model
            if self._scene_names is None:
                objects = {**domain.objects_dict, **domain.fixtures_dict}
                bodies = {name: model.body_name2id(obj.root_body) for name, obj in objects.items()}
                joints = []
                for name in model.joint_names:
                    if name.startswith(("robot", "gripper")):
                        continue
                    if model.jnt_type[model.joint_name2id(name)] in (2, 3):  # slide, hinge
                        joints.append((name, model.get_joint_qpos_addr(name)))
                self._scene_names = {"bodies": bodies, "joints": joints}
            names = self._scene_names
            return {
                "object_names": list(names["bodies"]),
                "object_pos": np.array(
                    [sim.data.body_xpos[i] for i in names["bodies"].values()], dtype=np.float32
                ),
                "articulation_names": [n for n, _ in names["joints"]],
                "articulation_qpos": np.array(
                    [sim.data.qpos[a] for _, a in names["joints"]], dtype=np.float32
                ),
            }
        except Exception as e:  # noqa: BLE001 -- recording must never abort a trial
            if self._scene_names is not False:
                logging.warning(f"scene state unavailable ({e!r}); mode recording off")
                self._scene_names = False
            return None

    def show_scene(self, trial_spec: TrialSpec | None = None):
        """Reset once and return the observation displayed to the operator."""
        kwargs = (
            {}
            if trial_spec is None
            else {"seed": trial_spec.env_seed, "options": {"init_state_id": trial_spec.init_state_id}}
        )
        observation, _ = self.vec_env.reset(**kwargs)
        self.view.stream.publish(_frame(observation))
        return observation

    def initial_state_hash(self) -> str:
        """Fingerprint the settled robot/object positions, velocities and actuator state."""
        data = self._sim().sim.data
        state = np.concatenate([data.qpos, data.qvel, data.act])
        state = np.round(state, 8).astype("<f8")
        state[state == 0] = 0  # canonicalize signed zero after rounding
        return hashlib.sha256(state.tobytes()).hexdigest()

    # ------------------------------------------------------------ operator perturbations

    def _status_extra(self) -> dict:
        return {
            "translation_noise": self.chain.readout.translation_noise,
            "corruption": self.chain.readout.label,
            "reversal_adapter": self.adapter.label,
        }

    def resolved_matrices(self) -> dict:
        """For run provenance: the matrices in force, as lists (None when off)."""
        m, f = self.chain.readout.matrix, self.adapter.matrix
        _, std = FlowControlSteering.action_mean_std(self.postprocessor)
        normalized = self.adapter.normalized_matrix(std[:N_ACTION_DIMS])
        curl = self.chain.readout.curl
        return {
            "corruption_matrix": matrix_list(m),
            "corruption_curl": None if curl is None else {"curl": curl.curl, "tau": curl.tau, "dt": curl.dt},
            "corruption_quantize": self.chain.readout.quantize,
            "reversal_adapter_matrix": matrix_list(f),
            "reversal_adapter_coordinates": "environment",
            "reversal_adapter_normalized_matrix": matrix_list(normalized),
            "reversal_adapter_modules": {"user": self.adapter.user_label, "task": self.adapter.task_label},
            "reversal_adapter_kind": self.adapter.kind,
        }

    # ------------------------------------------------------------ mode

    def set_mode(
        self,
        mode: str,
        lam: int | None = None,
        n_reversal_steps: int | None = None,
        tau: int | None = None,
        ic_gain: float | str | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode '{mode}' (expected one of {', '.join(MODES)})")
        if lam is not None:
            FlowControlSteering.check_steps(
                "lambda", lam, 0, self.policy.config.num_inference_steps, "Euler steps"
            )
            self.lam = lam
        if tau is not None:
            FlowControlSteering.check_steps("tau", tau, 1, self.policy.config.chunk_size, "chunk steps")
            self.tau = tau
        if ic_gain is not None:
            self.ic_gain = ic_gain  # validated by the wrapper
        if n_reversal_steps is not None:
            total = self.policy.config.num_inference_steps
            FlowReversalSteering.check_steps("n_reversal_steps", n_reversal_steps, 1, total, "Euler steps")
            self.n_reversal_steps = n_reversal_steps
        self.mode = mode
        if mode != "policy" and self.operator is None:
            self.chain.attach_spacemouse()

    def apply_control(self, control: ControlSettings) -> None:
        """Put the session in the state a `control:` block describes (mode, depths, noise, gain, matrices)."""
        self.chain.readout.quantize = False
        self.chain.readout.curl = None
        self.chain.readout.matrix = control.corruption_matrix
        self.chain.readout.curl = control.corruption_curl
        self.chain.readout.quantize = control.corruption_quantize
        self.chain.readout.label = spec_label(control.corruption)
        self.adapter.matrix = control.reversal_adapter_matrix
        self.adapter.label = spec_label(control.reversal_adapter)
        self.adapter.user = self.adapter.task = None
        self.adapter.user_label = self.adapter.task_label = None
        if control.reversal_adapter_user_path or control.reversal_adapter_task_path:
            self.adapter.load_modules(
                control.reversal_adapter_user_path,
                control.reversal_adapter_task_path,
                device=next(self.policy.parameters()).device,
            )
        self.tau, self.lam, self.ic_gain = control.tau, control.lam, control.ic_gain
        self.zeroed_dims = tuple(control.zeroed_dims)
        self.chain.readout.translation_noise = control.translation_noise
        self.chain.readout.rotation_noise = control.rotation_noise
        self.chain.readout.gripper_noise = control.gripper_noise
        self.translation_gain = (
            control.translation_gain
        )  # the setting (null, auto or a number); see refresh_translation_gain
        self.chain.gain.gain = control.resolved_translation_gain
        FlowReversalSteering.check_steps(
            "n_reversal_steps",
            control.n_reversal_steps,
            1,
            self.policy.config.num_inference_steps,
            "Euler steps",
            optional=True,
        )
        self.n_reversal_steps = control.n_reversal_steps  # None = full reversal, so set it explicitly
        self.pinned_steps = control.pinned_steps  # None = the executed prefix; the steering core checks it
        self.operator_dims = tuple(control.operator_dims)
        self.set_mode(control.mode)

    def refresh_translation_gain(self) -> str:
        """Re-derive an `auto` gain from the corruption now in force (the REPL swaps M live); returns a status line."""
        if self.translation_gain == "auto":
            m = self.chain.readout.matrix
            self.chain.gain.gain = gain_that_cannot_clip(1.0 if m is None else float(np.linalg.norm(m, 2)))
        gain = self.chain.gain.gain
        if gain is None:
            return "Translation gain: off (the corruption can clip the command)."
        auto = " (auto)" if self.translation_gain == "auto" else ""
        return f"Translation gain: {gain:g}{auto}: your command is limited to unit length and scaled by it."

    def mode_label(self) -> str:
        """Short label for the terminal and the live view."""
        if self.mode == "shared_flow_control":
            tau = self.tau or self.policy.config.chunk_size
            return f"{self.mode} tau={tau} lambda={self.lam} ic_gain={self.ic_gain}"
        if self.mode == "shared_flow_reversal_steering":
            return f"{self.mode} n={self.n_reversal_steps or self.policy.config.num_inference_steps}"
        return self.mode

    def announce_mode(self) -> str:
        """What the current mode does, for the terminal."""
        lines = []
        if self.mode != "policy":
            lines.append(
                "Keyboard (click the live view first): arrows = x/y, PgUp/PgDn or W/S = z, "
                "Space = gripper, hold Shift = full speed."
            )
        total = self.policy.config.num_inference_steps
        if self.mode == "teleop":
            lines.append("Teleop: you drive x/y/z and the gripper (a synthetic operator turns the hand too);")
            lines.append("the model is not involved.")
            lines.append("Press Enter to start a rollout.")
        elif self.mode == "shared_override":
            lines.append("Shared override: pi0.5 drives, your x/y/z replaces its translation.")
        elif self.mode == "shared_flow_control":
            chunk = self.policy.config.chunk_size
            tau = self.tau or chunk
            units = (
                "the checkpoint's action units"
                if self.ic_gain == "action"
                else f"{self.ic_gain} sigma per full push"
            )
            lines.append(
                f"Shared flow control (Kao et al.): your x/y/z, in {units}, is written into the first tau={tau} of "
                f"{chunk} chunk steps of x_t at the start of the first lambda={self.lam} of {total} denoising steps "
                "(idle input = pure policy)."
            )
        elif self.mode == "shared_flow_reversal_steering":
            n = total if self.n_reversal_steps is None else self.n_reversal_steps
            if n >= total:
                lines.append(
                    "Shared flow reversal steering: your x/y/z defines a reference chunk that is "
                    f"inverted through the flow ({total} reverse steps) to its noise; pi0.5 then "
                    "denoises from that noise (idle input = pure policy)."
                )
            else:
                lines.append(
                    "Shared flow reversal steering: your x/y/z defines a reference chunk that is "
                    f"inverted {n} of {total} steps (partway to noise, t={n / total:.1f}); pi0.5 then "
                    f"denoises from there in {n} steps (idle input = pure policy)."
                )
        return "\n".join(lines)

    def _rollout_kwargs(self) -> dict:
        source = self.chain.source
        if self.mode == "teleop":
            return {
                "policy": self._zero_policy,
                "env_preprocessor": _identity,
                "env_postprocessor": _identity,
                "preprocessor": _identity,
                "postprocessor": _identity,
                "action_hook": _teleop_hook(source, full=True),
                # a person gets teleop's longer limit; a synthetic operator the suite's, as in every other arm
                "max_steps": TELEOP_MAX_STEPS if self.operator is None else None,
            }
        kwargs = {
            "policy": self.policy,
            "env_preprocessor": self.env_preprocessor,
            "env_postprocessor": self.env_postprocessor,
            "preprocessor": self.preprocessor,
            "postprocessor": self.postprocessor,
            "action_hook": None,
            "max_steps": None,
        }
        if self.mode == "shared_override":
            kwargs["action_hook"] = _teleop_hook(source, full=False)
        elif self.mode in STEERING_MODES:
            kwargs["policy"] = self.steering_policy()
        return kwargs

    def steering_policy(self) -> SteeringPolicy:
        """lerobot's `SteeringPolicy` for the current shared mode and its knobs, over the loaded pi0.5 and its
        postprocessor, with this session's adapter in its adapter slot (the 7-D holder, read on every chunk, so
        a matrix or module swapped in the REPL takes effect at once). Built on first use and again when a knob
        changes. The command reaches it in the batch (`chunk_command`), as on the cursor."""
        cfg = self.policy.config
        fields = {
            "mode": STEERING_MODES[self.mode],
            "deadband": DEADBAND,
            "n_dims": 3,  # the commanded translation
            "base_policy_type": self.policy_cfg.type,
            "base_policy_path": str(self.settings.policy_path),
            "device": str(cfg.device),
        }
        if self.mode == "shared_flow_control":  # the command is written into the translation dims
            fields.update(
                tau=self.tau,
                lam=self.lam,
                ic_gain=self.ic_gain,
                zeroed_dims=list(self.zeroed_dims) or None,
                operator_dims=["translation"],
            )
        else:  # the reference pins a prefix (default: the executed steps); the rest is the policy's
            fields.update(
                n_reversal_steps=self.n_reversal_steps,
                pinned_steps=cfg.n_action_steps if self.pinned_steps is None else self.pinned_steps,
                operator_dims=list(self.operator_dims),
            )
        if self._steering is None or fields != self._steering_fields:
            steering = SteeringPolicy(
                SteeringConfig(**fields), base_policy=self.policy, postprocessor=self.postprocessor
            )
            steering.adapter = self.adapter
            _ = steering.core  # built now: its knobs are checked before the rollout starts
            self._steering, self._steering_fields = steering, fields
        return self._steering

    def chunk_command(self) -> torch.Tensor | None:
        """The operator's command for the chunk about to be sampled: one read of the teleop chain, (1, 3) in env
        units and float64 as read (the steering cores convert it; inside the deadband the chunk is the policy's
        own). None, with no read, when flow control writes nothing (lambda 0)."""
        if self.mode == "shared_flow_control" and not self.lam:
            return None
        return torch.as_tensor(np.asarray(self.chain.source.translation, dtype=np.float64))[None]

    def metrics(self) -> dict:
        """Steering statistics of the last rollout in the current mode."""
        steering = self._steering
        if steering is None or steering.config.mode != STEERING_MODES.get(self.mode):
            return {}
        core = steering.core
        if self.mode == "shared_flow_control":
            return {
                "guided_steps": int(core.guided_steps),
                "tau": core.tau,
                "lambda": core.lam,
                "ic_gain": core.ic_gain,
            }
        errors = core.reconstruction_errors
        return {
            "n_reversal_steps": core.n_reversal_steps or self.policy.config.num_inference_steps,
            "steered_chunks": int(core.steered_chunks),
            "reconstruction_error_mean": float(np.mean(errors)) if errors else None,
        }

    def stats_line(self) -> str:
        metrics = self.metrics()
        if self.mode == "shared_flow_control" and metrics:
            return f"flow guidance applied on {metrics['guided_steps']} denoising steps"
        if self.mode == "shared_flow_reversal_steering" and metrics:
            error = metrics["reconstruction_error_mean"]
            detail = f" (mean |executed - reference| translation: {error:.2f} std)" if error else ""
            return f"flow reversal steering applied on {metrics['steered_chunks']} chunks{detail}"
        return ""

    # ------------------------------------------------------------ rollout

    def rollout(
        self,
        prompt: str,
        recorder=None,
        max_steps: int | None = None,
        initial_observation=None,
        trial_spec: TrialSpec | None = None,
    ) -> RolloutResult:
        """One rollout in the current scene and mode, driven by `prompt`.

        `recorder(step=, observation=, action=, policy_action=, reward=, terminated=,
        truncated=, info=)` is called once per step with the observation the policy saw,
        the executed action and the policy's own plan for that step. `max_steps` overrides the suite's episode length (teleop mode has its
        own default).
        """
        k = self._rollout_kwargs()
        policy, action_hook = k["policy"], k["action_hook"]
        preprocessor, postprocessor = k["preprocessor"], k["postprocessor"]
        env_preprocessor, env_postprocessor = k["env_preprocessor"], k["env_postprocessor"]
        stream = self.view.stream
        vec_env = self.vec_env

        policy.reset()
        if self.operator is not None:
            self.operator.reset()
        self.chain.reset()  # a curl corruption starts from rest
        observation = initial_observation
        if observation is None:
            observation = self.show_scene(trial_spec)
        if trial_spec is not None:
            self.chain.readout.rng = np.random.default_rng(trial_spec.input_noise_seed)
        max_steps = max_steps or k["max_steps"] or int(vec_env.call("_max_episode_steps")[0])
        frames = [_frame(observation)]  # what show_scene has already published
        stream.set_status(
            prompt=prompt, step=0, max_steps=max_steps, state="running", action=None, mode=self.mode_label()
        )

        success = False
        step = 0
        while step < max_steps:
            step_start = time.time()
            self.chain.advance(step)  # the curl corruption's clock, before anything reads the source
            chunk_index = step // self.settings.n_action_steps
            if self.operator is not None:
                if self.operator.should_refresh(step):
                    if trial_spec is not None:
                        torch.manual_seed((trial_spec.operator_seed + self.operator.refreshes) % 2**32)
                    self._refresh_operator(observation, step=step)
                self.operator.advance(step)  # deliver before the wrappers sample the source
            obs = preprocess_observation(observation)
            obs["task"] = [prompt]
            obs = env_preprocessor(obs)
            obs = preprocessor(obs)
            new_chunk = step % self.settings.n_action_steps == 0
            if new_chunk and self.mode in STEERING_MODES:
                obs["command"] = self.chunk_command()  # read once per chunk and held, as the policy plans
            if trial_spec is not None and new_chunk:
                torch.manual_seed((trial_spec.policy_seed + chunk_index) % 2**32)
            with torch.inference_mode(), self.autocast_ctx:
                action = policy.select_action(obs)
            # The policy's own plan for this step: the steering policy exposes the chunk the
            # base would have produced without the operator; elsewhere it is the action
            # itself, before any operator hook.
            plan = getattr(policy, "last_policy_action", None)
            action_numpy = env_postprocessor({ACTION: postprocessor(action)})[ACTION].to("cpu").numpy()
            policy_numpy = (
                action_numpy
                if plan is None or plan is action
                else env_postprocessor({ACTION: postprocessor(plan)})[ACTION].to("cpu").numpy()
            )
            if action_hook is not None:
                action_numpy = action_hook(action_numpy)

            previous_observation = observation
            # Scene state *before* the step: the vector env auto-resets on termination, so the
            # post-step state of a final step would be the next episode's fresh scene.
            scene = self.scene_state() if recorder is not None else None
            observation, reward, terminated, truncated, info = vec_env.step(action_numpy)
            step += 1
            if recorder is not None:
                recorder(
                    step=step,
                    observation=previous_observation,
                    action=action_numpy,
                    policy_action=policy_numpy,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    scene=scene,
                )
            stream.set_status(step=step, action=action_numpy[0].tolist())

            if "final_info" in info:
                success = bool(info["final_info"]["is_success"][0])
            elif "is_success" in info:
                is_success = info["is_success"]
                success = bool(is_success[0] if hasattr(is_success, "__len__") else is_success)

            frames.append(_frame(observation))
            stream.publish(frames[-1])
            if terminated[0] or truncated[0]:
                break
            print(f"\r  step {step}/{max_steps}", end="", flush=True)

            leftover = 1.0 / RATE_HZ - (time.time() - step_start)
            if leftover > 0 and self.operator is None:  # real-time pacing only matters to a human
                time.sleep(leftover)

        print()
        stream.set_status(state="success" if success else "failed")
        return RolloutResult(success=success, steps=step, frames=frames, metrics=self.metrics())

    def policy_batch(self, observation, prompt: str) -> dict:
        """`observation` under `prompt` as the preprocessed batch the policy consumes (batch size 1)."""
        obs = preprocess_observation(observation)
        obs["task"] = [prompt]
        obs = self.env_preprocessor(obs)
        return self.preprocessor(obs)

    def policy_chunk(self, observation, prompt: str) -> torch.Tensor:
        """The policy's own action chunk for `observation` under `prompt`, in its normalized space."""
        obs = self.policy_batch(observation, prompt)
        with torch.inference_mode(), self.autocast_ctx:
            return self.policy.predict_action_chunk(obs)

    def env_actions(self, chunk: torch.Tensor) -> np.ndarray:
        """A normalized chunk (1, T, dim) as the env actions its steps would execute, (T, dim)."""
        steps = [
            self.env_postprocessor({ACTION: self.postprocessor(a)})[ACTION] for a in chunk.transpose(0, 1)
        ]
        return np.stack([a.to("cpu").numpy()[0] for a in steps])

    def _refresh_operator(self, observation, step: int = 0) -> None:
        """Ask the policy, prompted with the scene's real task, what it would do; hand that to the operator."""
        chunk = self.policy_chunk(observation, self.task_description)
        chunk = chunk[0, :, : self.operator_mean.shape[0]].float().cpu().numpy()
        self.operator.refresh(chunk * self.operator_std + self.operator_mean, step=step)  # env units

    def close(self) -> None:
        close_envs(self.envs_dict)
        self.view.close()
