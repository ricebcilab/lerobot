"""Project folders and checkpoint locations."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
CHECKPOINTS = ROOT.parent / "checkpoints"
RESULTS = ROOT.parent / "results"
REPORTS = ROOT / "reports"


def policy_path(name):
    """The policy checkpoint: its directory, a lerobot policy (config.json + model.safetensors and the policy's
    processor pipelines), which `policy.load_policy` reads; a cursor project's policy.pt in it is
    convert_checkpoints.py's to convert."""
    return CHECKPOINTS / "policies" / name


def policy_weights(name):
    """The file holding the policy's weights (for the session manifest's hash): model.safetensors."""
    return policy_path(name) / "model.safetensors"


def policy_log(name):
    return policy_path(name) / "train_log.json"


# The adapters' layout since 2026-11-25: cursor_sim/ the simulation's user and task modules (the studies play its
# user modules), cursor_participants/ the calibrated visits', cursor_time/ the demonstration-time curve's. An
# older run's records name a module in the layout before it; `current_name` finds it here.
MOVED = (("demo_time/image/", "cursor_time/"), ("demo_time/", "cursor_time/"), ("image/", "cursor_sim/"),
         ("participants/", "cursor_participants/"))


def current_name(name):
    """An adapter's name in the current layout (a current name stays as it is)."""
    for old, new in MOVED:
        if name.startswith(old):
            return new + name[len(old):]
    return f"cursor_sim/{name}" if "/" not in name and name.startswith("user_") else name


def adapter_dir(name):
    return CHECKPOINTS / "adapters" / current_name(name)


def adapter_path(name):
    return adapter_dir(name) / "adapter.pt"
