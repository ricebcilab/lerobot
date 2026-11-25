#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The steering package's action-space genericity: the config's ``reference_step`` picks FRS's held-command
reference on an action space wider than the 7-D operator space (pi0.5-DROID's 32 dims), flow control's written
dims are bounded by ``n_dims``, and the action spaces' own rules stay the default."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs import NormalizationMode
from lerobot.policies.steering import (
    GRIPPER_DIM,
    GRIPPER_OPEN,
    N_ACTION_DIMS,
    FlowControlSteering,
    FlowReversalSteering,
    SteeringConfig,
    SteeringPolicy,
)


def _mean_std(dims: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """The checkpoint's action normalization: MEAN_STD statistics over ``dims`` padded dims (7 of the env's)."""
    return 0.1 * np.arange(dims, dtype=np.float64), 0.5 + 0.25 * np.arange(dims, dtype=np.float64)


def _postprocessor(dims: int = 8) -> SimpleNamespace:
    """A postprocessor whose unnormalizer step carries the statistics ``action_mean_std`` reads."""
    mean, std = _mean_std(dims)
    step = SimpleNamespace(
        stats={"action": {"mean": mean, "std": std}},
        norm_map={"ACTION": NormalizationMode.MEAN_STD},
    )
    return SimpleNamespace(steps=[step])


class _Base:
    """A base policy offering pi0.5's sampler interface. Its chunk is a fixed tensor, so a write's touched dims are
    visible against ``initial``; the hooks it is given are applied as the sampler applies them (one Euler step)."""

    def __init__(
        self,
        *,
        chunk_size: int = 6,
        max_action_dim: int = 32,
        num_inference_steps: int = 4,
        n_action_steps: int = 3,
    ) -> None:
        self.config = SimpleNamespace(
            chunk_size=chunk_size,
            max_action_dim=max_action_dim,
            num_inference_steps=num_inference_steps,
            n_action_steps=n_action_steps,
            noise_time=1.0,
        )
        self.initial: torch.Tensor | None = None

    def reset(self) -> None:
        pass

    def velocity_field(self, batch: dict):
        """The contract's field: zero velocity, enough for a base that is measured but never integrated here."""
        return lambda x_t, t: torch.zeros_like(x_t)

    def predict_action_chunk(self, batch: dict, **hooks) -> torch.Tensor:
        g = torch.Generator().manual_seed(0)
        x = torch.rand(1, self.config.chunk_size, self.config.max_action_dim, generator=g)
        self.initial = x.clone()
        hook = hooks.get("x_t_hook")
        return x if hook is None else hook(0, 1.0, x)


def _policy(config: SteeringConfig, base: _Base, dims: int = 8) -> SteeringPolicy:
    """A steering policy over ``base`` and its action statistics."""
    return SteeringPolicy(config, base_policy=base, postprocessor=_postprocessor(dims))


def _normalized(values: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """The checkpoint's normalized action: ``(values - mean) / std`` over the values' own width."""
    width = values.shape[-1]
    return ((values.double() - torch.as_tensor(mean[:width])) / torch.as_tensor(std[:width])).to(values.dtype)


def test_frs_takes_the_configured_reference_step_on_a_wide_action_space():
    """The Franka plumbing's property: with ``reference_step`` a 7-wide command becomes the checkpoint's
    normalized action over dims 0-6 on any action space's width, and the padding beyond its own dims stays zero."""
    mean, std = _mean_std()
    base = _Base(max_action_dim=32)
    steering = _policy(SteeringConfig(mode="frs", n_dims=7, reference_step=1.0, device="cpu"), base)
    command = torch.tensor([0.4, -0.2, 0.1, 0.3, -0.5, 0.2, -0.1])
    reference = steering.core.reference_chunk(command)
    assert reference.shape == (1, base.config.chunk_size, 32)
    want = _normalized(command, mean, std)
    torch.testing.assert_close(reference[0, :, :N_ACTION_DIMS], want[None].expand(base.config.chunk_size, -1))
    assert not reference[..., N_ACTION_DIMS:].any()


def test_a_none_reference_step_keeps_the_action_spaces_own_rule():
    """``None`` is today's path: the 7-D operator space takes ``Flow.uniform_reference`` (translation held at the
    command in action units, rotation zero, the gripper at open), and a space without that gripper dim the held
    command at ``max_speed * dt``, bit for bit the explicit setting of that step."""
    mean, std = _mean_std()
    base = _Base(max_action_dim=8)
    steering = _policy(SteeringConfig(mode="frs", device="cpu"), base)
    command = torch.tensor([[0.5, -0.2, 0.1]])
    reference = steering.core.reference_chunk(command)
    mean_t = torch.as_tensor(mean[:N_ACTION_DIMS], dtype=torch.float64)
    std_t = torch.as_tensor(std[:N_ACTION_DIMS], dtype=torch.float64)
    want = torch.zeros(N_ACTION_DIMS, dtype=torch.float64)
    want[:3] = (command[0].double() - mean_t[:3]) / std_t[:3]
    want[3:6] = -mean_t[3:6] / std_t[3:6]
    want[GRIPPER_DIM] = (GRIPPER_OPEN - mean_t[GRIPPER_DIM]) / std_t[GRIPPER_DIM]
    want = want.to(command.dtype)
    torch.testing.assert_close(reference[0, :, :N_ACTION_DIMS], want[None].expand(base.config.chunk_size, -1))
    assert not reference[..., N_ACTION_DIMS:].any()

    cursor_command = torch.tensor([[0.5, -0.2]])
    step = SteeringConfig().max_speed * SteeringConfig().dt  # the cursor's reference rule
    default = _policy(SteeringConfig(mode="frs", device="cpu"), _Base(max_action_dim=2), dims=2)
    explicit = _policy(
        SteeringConfig(mode="frs", reference_step=step, device="cpu"), _Base(max_action_dim=2), 2
    )
    assert torch.equal(
        default.core.reference_chunk(cursor_command), explicit.core.reference_chunk(cursor_command)
    )


def test_a_nonpositive_reference_step_is_refused():
    for bad in (0, -0.5):
        with pytest.raises(ValueError, match="reference_step"):
            SteeringConfig(reference_step=bad)


def test_frs_delegates_the_dims_beyond_the_command():
    """The gripper depends on it: with a 7-wide command on a 32-wide action space, dims 0-6 are pinned for the
    reference's pinned prefix only, and dim 7 (the gripper) and the padding are the policy's at every step."""
    base = _Base(chunk_size=15, max_action_dim=32)
    steering = _policy(
        SteeringConfig(mode="frs", n_dims=7, operator_dims=None, pinned_steps=8, device="cpu"), base
    )
    delegated = steering.core._delegated
    assert delegated.shape == (15, 32)
    assert not delegated[:8, :7].any()  # the commanded dims, in the pinned executed prefix
    assert delegated[8:, :7].all()  # beyond the pinned prefix every dim is the policy's
    assert delegated[:, 7:].all()  # the gripper's dim and the padding never take the reference's zero


def test_command_dims_leaves_the_cursor_and_libero_masks_unchanged():
    """``command_dims`` only bites where the action space is wider than the command: the cursor's
    ``max_action_dim == n_dims == 2`` and LIBERO's named translation group build today's masks bit for bit."""
    cursor = _policy(SteeringConfig(mode="frs", device="cpu"), _Base(chunk_size=15, max_action_dim=2), dims=2)
    assert torch.equal(cursor.core._delegated, FlowReversalSteering.delegation_mask(15, 2, None, None))
    libero = _policy(
        SteeringConfig(mode="frs", n_dims=3, operator_dims=["translation"], pinned_steps=5, device="cpu"),
        _Base(chunk_size=15, max_action_dim=32),
    )
    assert torch.equal(
        libero.core._delegated, FlowReversalSteering.delegation_mask(15, 32, ("translation",), 5)
    )
    assert torch.equal(
        FlowReversalSteering.delegation_mask(15, 2, None, None, command_dims=2),
        FlowReversalSteering.delegation_mask(15, 2, None, None),
    )
    assert torch.equal(
        FlowReversalSteering.delegation_mask(15, 32, ("translation",), 5, command_dims=3),
        FlowReversalSteering.delegation_mask(15, 32, ("translation",), 5),
    )


def test_flow_control_writes_only_the_first_n_dims():
    """A 7-wide command on pi0.5-DROID's 32-wide action space writes dims 0-6 of ``x_t``; the gripper's dim and the
    padding keep what the chunk had (writing all 32 would be a broadcast error)."""
    base = _Base(max_action_dim=32)
    config = SteeringConfig(mode="flow_control", n_dims=7, device="cpu")
    steering = _policy(config, base)
    command = torch.tensor([[0.4, -0.2, 0.1, 0.3, -0.5, 0.2, -0.1]])
    chunk, _ = steering.core.steered_chunk({}, command, paired=False)
    torch.testing.assert_close(chunk[0, : config.tau, : config.n_dims], command.expand(config.tau, -1))
    assert torch.equal(chunk[..., config.n_dims :], base.initial[..., config.n_dims :])
    assert torch.equal(
        chunk[:, config.tau :, : config.n_dims], base.initial[:, config.tau :, : config.n_dims]
    )


def test_flow_control_keeps_the_named_operator_group():
    """``operator_dims=['translation']`` writes dims 0-2 as before, whatever ``n_dims`` is."""
    base = _Base(max_action_dim=32)
    config = SteeringConfig(mode="flow_control", n_dims=3, operator_dims=["translation"], device="cpu")
    steering = _policy(config, base)
    command = torch.tensor([[0.4, -0.2, 0.1]])
    chunk, _ = steering.core.steered_chunk({}, command, paired=False)
    torch.testing.assert_close(chunk[0, : config.tau, :3], command.expand(config.tau, -1))
    assert torch.equal(chunk[..., 3:], base.initial[..., 3:])


def test_the_cursor_action_spaces_written_region_is_the_old_slice_none():
    """``n_dims == max_action_dim == 2`` on the cursor: ``slice(0, n_dims)`` writes exactly what ``slice(None)``
    wrote, so the cursor's chunks are unchanged."""
    base = _Base(max_action_dim=2)
    config = SteeringConfig(mode="flow_control", device="cpu")
    steering = _policy(config, base, dims=2)
    command = torch.tensor([[0.5, -0.3]])
    chunk, _ = steering.core.steered_chunk({}, command, paired=False)
    old = base.initial.clone()
    FlowControlSteering.write(old, 0, command[:, None], config.tau, config.lam, dims=slice(None))
    assert torch.equal(chunk, old)


def test_the_existing_steering_defaults_still_validate():
    """The config's mode and its three knobs keep their defaults, the three modes still build, and the new field is
    off by default (the action space's own rule)."""
    config = SteeringConfig(device="cpu")
    assert config.mode == "frs"
    assert (config.tau, config.lam, config.ic_gain) == (5, 1, 1.0)
    assert config.reference_step is None
    for mode in ("none", "flow_control", "frs"):
        assert SteeringConfig(mode=mode, device="cpu").mode == mode
