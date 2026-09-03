"""A loaded policy, a LIBERO scene, the operator's input and the live view, in one object.

Both entry points drive a `Session`: the REPL changes its scene and mode
between rollouts, the experiment driver walks a schedule of scenes. `rollout`
is the step loop of `lerobot_eval.rollout()` with the user's prompt in place
of the env's task description, an optional action hook (teleop modes) and an
optional per-step recorder (experiments).
"""

import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch
from config import MODES, ControlSettings, SessionSettings, spec_label
from live_view import ACTION_LABELS, LiveView
from teleop import KeyboardReader, TeleopChain

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
    FlowControlPolicy,
    FlowReversalSteeringPolicy,
    ReversalAdapter,
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
        return torch.zeros(1, len(ACTION_LABELS))


def _identity(x):
    return x


def _teleop_hook(source, paste_gripper: bool) -> Callable[[np.ndarray], np.ndarray]:
    """Paste the teleop translation (and, for teleop mode, gripper) into the env action."""

    def hook(action: np.ndarray) -> np.ndarray:
        action = action.copy()
        action[0, :3] = source.translation
        if paste_gripper:
            action[0, 6] = source.gripper
        return action

    return hook


class Session:
    def __init__(self, settings: SessionSettings):
        self.settings = settings
        control = settings.control

        self.keyboard = KeyboardReader()
        self.chain = TeleopChain(self.keyboard, input_noise=control.input_noise)
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

        self.suite = self.task_id = None
        self.envs_dict = self.vec_env = None
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

        self._zero_policy = _ZeroPolicy()
        self._flow_policy: FlowControlPolicy | None = None
        self._frs_policy: FlowReversalSteeringPolicy | None = None
        self.mode = "policy"
        self.n_guided_steps = control.n_guided_steps
        self.n_reversal_steps = control.n_reversal_steps
        self.apply_control(control)

    @classmethod
    def from_settings(cls, settings: SessionSettings) -> "Session":
        return cls(settings)

    # ------------------------------------------------------------ scene

    @staticmethod
    def list_tasks(suite: str) -> list[str]:
        from libero.libero import benchmark

        bench = benchmark.get_benchmark_dict()
        if suite not in bench:
            raise ValueError(f"Unknown suite '{suite}'. Available: {', '.join(sorted(bench))}")
        return [t.language for t in bench[suite]().tasks]

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
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=self.policy_cfg
        )

    def show_scene(self) -> None:
        """Reset the env and put its first frame on the live view."""
        self.vec_env.reset()
        self.view.stream.publish(self.vec_env.envs[0].render())

    # ------------------------------------------------------------ operator perturbations

    def set_corruption(self, matrix, label: str | None) -> None:
        self.chain.corruption.matrix = matrix
        self.chain.corruption.label = label

    def set_reversal_adapter(self, matrix, label: str | None) -> None:
        self.adapter.matrix = matrix
        self.adapter.label = label

    def _status_extra(self) -> dict:
        corruption, adapter = self.chain.corruption, self.adapter
        return {
            "input_noise": self.chain.noisy.input_noise,
            "corruption": corruption.label if corruption.matrix is not None else None,
            "flow_adapter": adapter.label if adapter.matrix is not None else None,
        }

    def resolved_matrices(self) -> dict:
        """For run provenance: the matrices in force, as lists (None when off)."""
        m, f = self.chain.corruption.matrix, self.adapter.matrix
        return {
            "corruption_matrix": None if m is None else m.tolist(),
            "reversal_adapter_matrix": None if f is None else f.tolist(),
        }

    # ------------------------------------------------------------ mode

    def set_mode(
        self, mode: str, n_guided_steps: int | None = None, n_reversal_steps: int | None = None
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode '{mode}' (expected one of {', '.join(MODES)})")
        total = self.policy.config.num_inference_steps
        if n_guided_steps is not None:
            if not isinstance(n_guided_steps, int) or not 0 <= n_guided_steps <= total:
                raise ValueError(f"n_guided_steps must be an integer in [0, {total}] (denoising steps)")
            self.n_guided_steps = n_guided_steps
        if n_reversal_steps is not None:
            if not isinstance(n_reversal_steps, int) or not 1 <= n_reversal_steps <= total:
                raise ValueError(f"n_reversal_steps must be an integer in [1, {total}]")
            self.n_reversal_steps = n_reversal_steps
        self.mode = mode
        if mode != "policy":
            self.chain.attach_spacemouse()

    def apply_control(self, control: ControlSettings) -> None:
        """Put the session in the state a `control:` block describes (mode, depths, noise, matrices)."""
        self.set_corruption(control.corruption_matrix, spec_label(control.corruption))
        self.set_reversal_adapter(control.reversal_adapter_matrix, spec_label(control.reversal_adapter))
        self.chain.noisy.input_noise = control.input_noise
        total = self.policy.config.num_inference_steps
        if control.n_reversal_steps is not None and not 1 <= control.n_reversal_steps <= total:
            raise ValueError(f"n_reversal_steps must be an integer in [1, {total}]")
        self.n_reversal_steps = control.n_reversal_steps  # None = full reversal, so set it explicitly
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
                    f"denoises from there in {total - n} steps (idle input = pure policy)."
                )
            if self.adapter.matrix is not None:
                lines.append(self.adapter.describe())
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
                    self.policy, source, self.postprocessor, adapter=self.adapter
                )
            self._frs_policy.n_reversal_steps = self.n_reversal_steps
            kwargs["policy"] = self._frs_policy
        return kwargs

    def metrics(self) -> dict:
        """Steering statistics of the last rollout in the current mode."""
        if self.mode == "shared_flow_control" and self._flow_policy is not None:
            return {"guided_steps": int(self._flow_policy.guided_steps)}
        if self.mode == "shared_flow_reversal_steering" and self._frs_policy is not None:
            errors = self._frs_policy.reconstruction_errors
            total = self.policy.config.num_inference_steps
            return {
                "n_reversal_steps": self._frs_policy.n_reversal_steps or total,
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

    def rollout(self, prompt: str, recorder=None, max_steps: int | None = None) -> RolloutResult:
        """One rollout in the current scene and mode, driven by `prompt`.

        `recorder(step=, observation=, action=, reward=, terminated=, truncated=, info=)`
        is called once per step with the observation the policy saw and the executed
        action. `max_steps` overrides the suite's episode length (teleop mode has its
        own default).
        """
        k = self._rollout_kwargs()
        policy, action_hook = k["policy"], k["action_hook"]
        preprocessor, postprocessor = k["preprocessor"], k["postprocessor"]
        env_preprocessor, env_postprocessor = k["env_preprocessor"], k["env_postprocessor"]
        stream = self.view.stream
        vec_env = self.vec_env

        policy.reset()
        observation, _ = vec_env.reset()
        max_steps = max_steps or k["max_steps"] or int(vec_env.call("_max_episode_steps")[0])
        frames = [vec_env.envs[0].render()]
        stream.publish(frames[-1])
        stream.set_status(
            prompt=prompt, step=0, max_steps=max_steps, state="running", action=None, mode=self.mode_label()
        )

        success = False
        step = 0
        while step < max_steps:
            step_start = time.time()
            obs = preprocess_observation(observation)
            obs["task"] = [prompt]
            obs = env_preprocessor(obs)
            obs = preprocessor(obs)
            with torch.inference_mode(), self.autocast_ctx:
                action = policy.select_action(obs)
            action = postprocessor(action)
            transition = env_postprocessor({ACTION: action})
            action_numpy = transition[ACTION].to("cpu").numpy()
            if action_hook is not None:
                action_numpy = action_hook(action_numpy)

            previous_observation = observation
            observation, reward, terminated, truncated, info = vec_env.step(action_numpy)
            step += 1
            if recorder is not None:
                recorder(
                    step=step,
                    observation=previous_observation,
                    action=action_numpy,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            stream.set_status(step=step, action=action_numpy[0].tolist())

            if "final_info" in info:
                success = bool(info["final_info"]["is_success"][0])
            elif "is_success" in info:
                is_success = info["is_success"]
                success = bool(is_success[0] if hasattr(is_success, "__len__") else is_success)

            if terminated[0] or truncated[0]:
                # The sub-env auto-resets on termination, so render() would show a
                # fresh scene; use the final observation's agentview pixels instead
                # (flipped to match render() orientation).
                final = observation["pixels"]["image"][0][::-1, ::-1]
                frames.append(final)
                stream.publish(final)
                break

            frames.append(vec_env.envs[0].render())
            stream.publish(frames[-1])
            print(f"\r  step {step}/{max_steps}", end="", flush=True)

            leftover = 1.0 / RATE_HZ - (time.time() - step_start)
            if leftover > 0:
                time.sleep(leftover)

        print()
        stream.set_status(state="success" if success else "failed")
        return RolloutResult(success=success, steps=step, frames=frames, metrics=self.metrics())

    def close(self) -> None:
        close_envs(self.envs_dict)
        self.view.close()
