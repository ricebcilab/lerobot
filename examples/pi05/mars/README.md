# MARS: modular adapters for reversal steering

MARS (Modular Adapters for Reversal Steering) is shared autonomy with a generative
policy: a person's low-dimensional command (a SpaceMouse, a keyboard or a
simulated agent) steers a flow-matching policy through Flow Reversal Steering
(FRS), and a small learned adapter reverses the user's corruption of that
command and the task's obstacles, as a user module and a task module that can be
trained separately and composed. Two tasks carry the project. [`cursor/`](cursor/)
is a 2D cursor-reaching task with a small flow policy, a human study and a
simulated users x tasks grid. [`libero/`](libero/) steers the pi0.5 policy on
LIBERO manipulation with the same steering methods
(`lerobot.policies.pi05.steering`).

Names: MARS is the method; `cursor_sa` is the cursor example's package, `cursor_flow`
its policy in `lerobot.policies`, and `ricebcilab/mars-cursor` the Hub repo of its
checkpoints; FRS is LIBERO's `shared_flow_reversal_steering` mode (the cursor's `frs`),
and MARS is FRS with a reversal adapter.

## Quick start (cursor task)

Run from the repo root.

```bash
uv sync --extra mars        # or: pip install -e ".[mars]"
uv run --extra mars python examples/pi05/mars/cursor/hub.py download
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study.yaml --participant p1
uv run --extra mars python examples/pi05/mars/cursor/analyze_results.py --participants p1
```

`hub.py download` fetches the trained checkpoints (the folders in
`cursor/hub_manifest.yaml`, about 97 MB, from the public Hub repo
`ricebcilab/mars-cursor`; no login needed), `run_experiment.py` runs a study
session for participant `p1` (SpaceMouse and a display; use `--participant sim`
for the expert agent, headless, which needs neither), and `analyze_results.py`
prints the per-arm table and writes that participant's tables and figures.
Results go to `examples/pi05/mars/results/`, checkpoints to
`examples/pi05/mars/checkpoints/` (both gitignored).

The cursor trainers write the cursor project's checkpoint formats (`policy.pt`,
`adapter.pt`) and the loaders read both; rerun
`cursor/convert_checkpoints.py` after training to get the lerobot format. When a
policy folder holds both, `policy.pt` wins, with a warning
([cursor/README.md](cursor/README.md#training)).

Tests: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor -q`. lerobot's
`tests/policies/pi0_pi05` loads the gated `google/paligemma-3b-pt-224` and needs a
Hugging Face token that has accepted its license (`hf auth login` or
`HF_TOKEN`); without one those tests fail with 401 errors.

## SpaceMouse

Both tasks read a 3Dconnexion SpaceMouse through lerobot's `spacemouse`
teleoperator (`lerobot.teleoperators.spacemouse`), straight from Linux
`/dev/hidraw`; no driver or extra package is needed. Any 3Dconnexion device is
found: vendor `256f`, or vendor `046d` (Logitech) with a `c6xx` product, so a
Logitech keyboard or mouse is never picked. To pin one device, set the
`SPACEMOUSE_HID_ID` environment variable to its HID id, as in
`/sys/class/hidraw/*/device/uevent` (e.g. `export SPACEMOUSE_HID_ID=0000046D:0000C62B`);
both the cursor app and the LIBERO runners read it. Unset, any 3Dconnexion device
is taken. Check that it is found:

```bash
uv run --extra mars python -c "from lerobot.teleoperators.spacemouse.teleop_spacemouse import find_spacemouse; print(find_spacemouse())"
```

The `/dev/hidraw*` node must be readable by your user. A udev rule, e.g. in
`/etc/udev/rules.d/99-spacemouse.rules` (udev match values take `|`
alternatives):

```
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0666"
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c6*", MODE="0666"
```

or the group form, for users in `plugdev`:

```
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", GROUP="plugdev", MODE="0660"
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c6*", GROUP="plugdev", MODE="0660"
```

The one-line `ATTRS{idVendor}=="046d|256f"` also works, but it opens every
Logitech HID device (keyboards included) to all users; keep the `c6*` product
match on `046d`. Then `sudo udevadm control --reload-rules && sudo udevadm trigger`
and replug. A Bluetooth-paired SpaceMouse has no USB parent, so the `ATTRS{idVendor}`
match does not apply to it; `TAG+="uaccess"` in place of `MODE`/`GROUP` (seat-based
access for the logged-in user) is an alternative.

The full-scale count (default 350 raw counts) maps a full push to 1.0; set the
`SPACEMOUSE_FULL_SCALE` environment variable to change it in both apps
(e.g. `export SPACEMOUSE_FULL_SCALE=700`). To check it, push the stick fully
along one axis and watch that axis read about 1.0 (clipped at 1.0; a value well
short of 1.0 at full push means the full-scale count is too large):

```bash
uv run --extra mars python -c "
import time
from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseReader, find_spacemouse
r = SpaceMouseReader(find_spacemouse())
while True: print(r.axes, r.buttons); time.sleep(0.2)"
```

## Where things are

- [`cursor/README.md`](cursor/README.md): the cursor task, its experiment portal, training, sweeps, reports and tests.
- [`libero/README.md`](libero/README.md): the LIBERO example (evaluate, drive and study steering of pi0.5), started with `./examples/pi05/mars/libero/run.sh setup`.
- [`figures/`](figures/): the paper's three figures and `build_figures.py`. Fig 1's
  numbers in `figures/stats.json` reproduce with `build_figures.py --device cpu`, the
  device they were built on; CUDA draws different policy noise.
