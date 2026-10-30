"""Headless 2D cursor-reaching task with a variable number of targets: lerobot's
`lerobot.envs.cursor`, re-exported so the example's imports stay as they were."""

from lerobot.envs.cursor import *  # noqa: F403
from lerobot.envs.cursor import _layout  # noqa: F401  (tests exercise the layout sampler directly)
