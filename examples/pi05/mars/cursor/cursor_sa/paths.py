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


def adapter_dir(name):
    return CHECKPOINTS / "adapters" / name


def adapter_path(name):
    return adapter_dir(name) / "adapter.pt"
