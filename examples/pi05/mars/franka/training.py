"""The one MARS training loop: AdamW + cosine, gradients clipped, non-finite batches skipped, a log.

Phase B's user-module trainer and Phase C's task-module trainer train different modules on different
recipes, but the loop around a module is the same three lines of torch and one decision: a batch whose
loss or gradient is not finite does not update the weights. It used to live in ``train_user`` (with
copies in the LIBERO and cursor trainers); both trainers here import it from this module so there is one
copy, and the rule that a skipped batch leaves the weights whole is written down once.

The two functions are the trainers' whole contract: ``train_module`` owns the optimizer, the schedule,
the window and the held-out call; ``step_if_finite`` is the one optimizer step it takes. ``batch_loss(step)``
returns one batch's loss (the module is trained through whatever forward computed it) and ``validate()``
returns the named held-out numbers, or None.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable

import torch
from torch import nn


def step_if_finite(
    opt: torch.optim.Optimizer,
    loss: torch.Tensor,
    params: Iterable[nn.Parameter],
    clip: float = 1.0,
) -> bool:
    """One optimizer step, unless the loss or the clipped gradient norm is not finite (a batch where a flow
    blew up): then the step is skipped and the weights stay whole. Returns whether a step was taken."""
    opt.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        return False
    loss.backward()
    norm = nn.utils.clip_grad_norm_(list(params), clip)
    if not torch.isfinite(norm):
        opt.zero_grad(set_to_none=True)
        return False
    opt.step()
    return True


def train_module(
    module: nn.Module,
    steps: int,
    lr: float,
    batch_loss: Callable[[int], torch.Tensor],
    validate: Callable[[], dict] | None = None,
    log_every: int = 1000,
    weight_decay: float = 1e-4,
    clip: float = 1.0,
) -> list[dict]:
    """A MARS module's training loop: AdamW with a cosine schedule over ``steps``, gradients clipped to
    ``clip``, non-finite batches skipped (``step_if_finite``). ``batch_loss(step)`` returns one batch's
    loss. Every ``log_every`` steps, and at the last, the module is put in eval mode for ``validate()``
    (named numbers, or None) and {step, train_loss (the window's mean), skipped, **validation} is logged and
    printed. Returns the log."""
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    log: list[dict] = []
    running, skipped, tic = 0.0, 0, time.time()
    for step in range(1, steps + 1):
        loss = batch_loss(step)
        skipped += not step_if_finite(opt, loss, module.parameters(), clip)
        sched.step()
        running += loss.item() if torch.isfinite(loss) else 0.0
        if step % log_every == 0 or step == steps:
            module.eval()
            val = validate() if validate is not None else {}
            module.train()
            entry = {"step": step, "train_loss": running / log_every, "skipped": skipped, **val}
            log.append(entry)
            shown = "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in val.items())
            print(
                f"step {step:6d}  train {entry['train_loss']:.4f}  {shown}"
                + (f"  skipped {skipped}" if skipped else "")
                + f"  ({time.time() - tic:.0f}s)",
                flush=True,
            )
            running = 0.0
    return log
