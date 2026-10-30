"""Project folders and checkpoint locations."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
CHECKPOINTS = ROOT.parent / "checkpoints"
RESULTS = ROOT.parent / "results"
REPORTS = ROOT / "reports"


def policy_dir(name):
    return CHECKPOINTS / "policies" / name


def policy_path(name):
    """The policy checkpoint: its directory, a lerobot policy (config.json + model.safetensors, see
    convert_checkpoints.py); FlowPolicy.load reads it, and a trainer's unconverted policy.pt in it."""
    return policy_dir(name)


def policy_weights(name):
    """The file holding the policy's weights (for the session manifest's hash): an unconverted policy.pt
    if there is one (FlowPolicy.load reads it first), else model.safetensors."""
    pt = policy_dir(name) / "policy.pt"
    return pt if pt.is_file() else policy_dir(name) / "model.safetensors"


def policy_log(name):
    return policy_dir(name) / "train_log.json"


def adapter_dir(name):
    return CHECKPOINTS / "adapters" / name


def adapter_path(name):
    return adapter_dir(name) / "adapter.pt"
