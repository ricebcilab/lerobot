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
from config import MODES, ControlSettings, SessionSettings, spec_label
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
from lerobot.policies.pi05.steering import (
    GRIPPER_DIM,
    N_ACTION_DIMS,
    FlowControlPolicy,
    FlowReversalSteeringPolicy,
    ReversalAdapter,
    get_action_mean_std,
)
from lerobot.utils.constants import ACTION

VIDEO_FPS = 30
RATE_HZ = 20  # LIBERO control rate; caps the loop so teleop feels real-time
# Suite defaults (~14 s) are too short for manual driving. Must stay below
# robosuite's horizon of 1000 (which includes the 10 settle steps at reset):
# LIBERO's step() replaces the horizon `done` with task success, so exceeding
# the horizon raises "executing action in terminated episode" instead of ending.
TELEOP_MAX_STEPS = 900


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


def _teleop_hook(source, paste_gripper: bool) -> Callable[[np.ndarray], np.ndarray]:
    """Paste the teleop translation (and, for teleop mode, gripper) into the env action."""

    def hook(action: np.ndarray) -> np.ndarray:
        action = action.copy()
        action[0, :3] = source.translation
        if paste_gripper:
            action[0, GRIPPER_DIM] = source.gripper
        return action

    return hook


class Session:
    def __init__(self, settings: SessionSettings):
        self.settings = settings
        control = settings.control

        self.keyboard = KeyboardReader()
        self.chain = TeleopChain(self.keyboard, input_noise=control.input_noise)
        # A synthetic operator (the task-prompted policy) replaces the SpaceMouse/keyboard.
        self.operator: PolicyOperator | None = None
        if settings.operator == "policy":
            self.operator = PolicyOperator(settings.n_action_steps, settings.policy_operator)
            self.chain.attach_operator(self.operator)
        self.adapter = ReversalAdapter()
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
            self.operator_mean, self.operator_std = get_action_mean_std(self.postprocessor)

        self._zero_policy = _ZeroPolicy()
        self._flow_policy: FlowControlPolicy | None = None
        self._frs_policy: FlowReversalSteeringPolicy | None = None
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
            "input_noise": self.chain.noisy.input_noise,
            "corruption": self.chain.corruption.label,
            "reversal_adapter": self.adapter.label,
        }

    def resolved_matrices(self) -> dict:
        """For run provenance: the matrices in force, as lists (None when off)."""
        m, f = self.chain.corruption.matrix, self.adapter.matrix
        _, std = get_action_mean_std(self.postprocessor)
        normalized = self.adapter.normalized_matrix(std[:N_ACTION_DIMS])
        return {
            "corruption_matrix": None if m is None else m.tolist(),
            "reversal_adapter_matrix": None if f is None else f.tolist(),
            "reversal_adapter_coordinates": "environment",
            "reversal_adapter_normalized_matrix": None if normalized is None else normalized.tolist(),
        }

    # ------------------------------------------------------------ mode

    def _check_depth(self, name: str, value, low: int) -> None:
        total = self.policy.config.num_inference_steps
        if not isinstance(value, int) or not low <= value <= total:
            raise ValueError(f"{name} must be an integer in [{low}, {total}] (denoising steps)")

    def set_mode(
        self, mode: str, n_guided_steps: int | None = None, n_reversal_steps: int | None = None
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode '{mode}' (expected one of {', '.join(MODES)})")
        if n_guided_steps is not None:
            self._check_depth("n_guided_steps", n_guided_steps, 0)
            self.n_guided_steps = n_guided_steps
        if n_reversal_steps is not None:
            self._check_depth("n_reversal_steps", n_reversal_steps, 1)
            self.n_reversal_steps = n_reversal_steps
        self.mode = mode
        if mode != "policy" and self.operator is None:
            self.chain.attach_spacemouse()

    def apply_control(self, control: ControlSettings) -> None:
        """Put the session in the state a `control:` block describes (mode, depths, noise, matrices)."""
        self.chain.corruption.matrix = control.corruption_matrix
        self.chain.corruption.label = spec_label(control.corruption)
        self.adapter.matrix = control.reversal_adapter_matrix
        self.adapter.label = spec_label(control.reversal_adapter)
        self.chain.noisy.input_noise = control.input_noise
        if control.n_reversal_steps is not None:
            self._check_depth("n_reversal_steps", control.n_reversal_steps, 1)
        self.n_reversal_steps = control.n_reversal_steps  # None = full reversal, so set it explicitly
        self.operator_dims = tuple(control.operator_dims)
        if self._frs_policy is not None:
            self._frs_policy.operator_dims = self.operator_dims
        self.set_mode(control.mode, control.n_guided_steps)

    def mode_label(self) -> str:
        """Short label for the terminal and the live view."""
        if self.mode == "shared_flow_control":
            return f"{self.mode} n_guided_steps={self.n_guided_steps}"
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
            lines.append("Teleop: you drive x/y/z and the gripper; the model is not involved.")
            lines.append("Press Enter to start a rollout.")
        elif self.mode == "shared_override":
            lines.append("Shared override: pi0.5 drives, your x/y/z replaces its translation.")
        elif self.mode == "shared_flow_control":
            lines.append(
                f"Shared flow control: your x/y/z steers the first n_guided_steps={self.n_guided_steps} of "
                f"{total} denoising steps (idle input = pure policy)."
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
                "action_hook": _teleop_hook(source, paste_gripper=True),
                "max_steps": TELEOP_MAX_STEPS,
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
            kwargs["action_hook"] = _teleop_hook(source, paste_gripper=False)
        elif self.mode == "shared_flow_control":
            if self._flow_policy is None:
                self._flow_policy = FlowControlPolicy(
                    self.policy, source, self.n_guided_steps, self.postprocessor
                )
            self._flow_policy.n_guided_steps = self.n_guided_steps
            kwargs["policy"] = self._flow_policy
        elif self.mode == "shared_flow_reversal_steering":
            if self._frs_policy is None:
                self._frs_policy = FlowReversalSteeringPolicy(
                    self.policy,
                    source,
                    self.postprocessor,
                    adapter=self.adapter,
                    operator_dims=self.operator_dims,
                )
            self._frs_policy.n_reversal_steps = self.n_reversal_steps
            self._frs_policy.operator_dims = self.operator_dims
            kwargs["policy"] = self._frs_policy
        return kwargs

    def metrics(self) -> dict:
        """Steering statistics of the last rollout in the current mode."""
        if self.mode == "shared_flow_control" and self._flow_policy is not None:
            return {"guided_steps": int(self._flow_policy.guided_steps)}
        if self.mode == "shared_flow_reversal_steering" and self._frs_policy is not None:
            errors = self._frs_policy.reconstruction_errors
            return {
                "n_reversal_steps": self._frs_policy.n_reversal_steps
                or self.policy.config.num_inference_steps,
                "steered_chunks": int(self._frs_policy.steered_chunks),
                "reconstruction_error_mean": float(np.mean(errors)) if errors else None,
            }
        return {}

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
        observation = initial_observation
        if observation is None:
            observation = self.show_scene(trial_spec)
        if trial_spec is not None:
            self.chain.noisy._rng = np.random.default_rng(trial_spec.input_noise_seed)
        max_steps = max_steps or k["max_steps"] or int(vec_env.call("_max_episode_steps")[0])
        frames = [_frame(observation)]  # what show_scene has already published
        stream.set_status(
            prompt=prompt, step=0, max_steps=max_steps, state="running", action=None, mode=self.mode_label()
        )

        success = False
        step = 0
        while step < max_steps:
            step_start = time.time()
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
            if trial_spec is not None and step % self.settings.n_action_steps == 0:
                torch.manual_seed((trial_spec.policy_seed + chunk_index) % 2**32)
            with torch.inference_mode(), self.autocast_ctx:
                action = policy.select_action(obs)
            # The policy's own plan for this step: the steering wrappers expose the chunk
            # they would have produced without the operator; elsewhere it is the action
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

    def policy_chunk(self, observation, prompt: str) -> torch.Tensor:
        """The policy's own action chunk for `observation` under `prompt`, in its normalized space."""
        obs = preprocess_observation(observation)
        obs["task"] = [prompt]
        obs = self.env_preprocessor(obs)
        obs = self.preprocessor(obs)
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
