#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseConfig(TeleoperatorConfig):
    """A 3Dconnexion SpaceMouse read from /dev/hidraw (Linux).

    `hid_id` pins one device by a substring of its sysfs uevent, e.g. "0000046D:0000C62B";
    None takes the first 3Dconnexion device found. `full_scale` is the raw count of a full
    push, which reads 1.0.
    """

    hid_id: str | None = None
    full_scale: float = 350.0
