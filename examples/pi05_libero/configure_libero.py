#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Configure hf-libero non-interactively with project-local assets."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import yaml


ASSET_SUBDIRECTORIES = (
    "articulated_objects",
    "stable_scanned_objects",
    "turbosquid_objects",
    "stable_hope_objects",
    "scenes",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=repo_root / ".cache" / "libero",
        help="LIBERO config/data directory (default: <repo>/.cache/libero)",
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Write the config without downloading simulator assets.",
    )
    return parser.parse_args()


def find_libero_root() -> Path:
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("hf-libero is not installed; run setup.sh first")

    package_root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    if not package_root.is_dir():
        raise RuntimeError(f"Could not locate the hf-libero package data at {package_root}")
    return package_root.resolve()


def write_config(config_dir: Path, libero_root: Path, assets_dir: Path) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = config_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    config = {
        "benchmark_root": str(libero_root),
        "bddl_files": str(libero_root / "bddl_files"),
        "init_states": str(libero_root / "init_files"),
        "datasets": str(datasets_dir),
        "assets": str(assets_dir),
    }
    config_file = config_dir / "config.yaml"
    config_file.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return config_file


def assets_are_complete(assets_dir: Path) -> bool:
    return all((assets_dir / name).is_dir() for name in ASSET_SUBDIRECTORIES)


def download_assets(assets_dir: Path) -> None:
    # The config must exist before this import, otherwise hf-libero prompts on stdin.
    from libero.libero.utils.download_utils import download_assets_from_huggingface

    download_assets_from_huggingface(download_dir=str(assets_dir))
    if not assets_are_complete(assets_dir):
        missing = [name for name in ASSET_SUBDIRECTORIES if not (assets_dir / name).is_dir()]
        raise RuntimeError(f"LIBERO asset download is incomplete; missing: {', '.join(missing)}")


def link_assets(libero_root: Path, assets_dir: Path) -> Path:
    """Make hf-libero's hard-coded package-local asset lookup use our cache."""
    package_assets = libero_root / "assets"
    if package_assets.is_symlink():
        if package_assets.resolve() != assets_dir.resolve():
            raise RuntimeError(
                f"Refusing to replace existing LIBERO asset link {package_assets} -> "
                f"{package_assets.resolve()}"
            )
        return package_assets
    if package_assets.exists():
        raise RuntimeError(f"Refusing to replace existing LIBERO assets at {package_assets}")

    package_assets.symlink_to(assets_dir.resolve(), target_is_directory=True)
    return package_assets


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.expanduser().resolve()
    assets_dir = config_dir / "assets"
    libero_root = find_libero_root()
    config_file = write_config(config_dir, libero_root, assets_dir)

    # Ensure imports in the current process use the config we just wrote.
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    if args.skip_assets:
        print(f"Wrote LIBERO config: {config_file}")
        print("Skipped simulator assets; imports work, but environments cannot reset yet.")
        return

    if not assets_are_complete(assets_dir):
        download_assets(assets_dir)
    package_assets = link_assets(libero_root, assets_dir)

    print(f"Wrote LIBERO config: {config_file}")
    print(f"LIBERO assets: {assets_dir}")
    print(f"Package asset link: {package_assets} -> {package_assets.resolve()}")


if __name__ == "__main__":
    main()

