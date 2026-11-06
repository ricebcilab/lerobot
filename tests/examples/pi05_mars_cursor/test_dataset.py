"""The scene dataset (cursor_sa/dataset.py, `SceneDataset`): a layout's tokens encoded once and shared by every draw that
holds the layout, gathered bit for bit, and an append that crashes leaving the index consistent."""

import json

import numpy as np
import pytest
import torch
from _view_stub import TokenStub
from cursor_sa.dataset import SceneDataset, encoder_digest, layout_digests
from cursor_sa.env import Layouts, TaskConfig, move_layouts, obstacle_spec, sample_layouts
from train import train_adapter

from lerobot.policies.steering import scene_for

SIZE = 32
IMAGE = "{kind: disc, radius: 0.08, along: 0.5}"


def _layouts(seed, n=5):
    task = TaskConfig(obstacle=obstacle_spec({"kind": "disc", "radius": 0.08, "along": 0.5}))
    return sample_layouts(np.random.default_rng(seed), n, (4, 4), task)


def _store(tmp_path, digest=None):
    return SceneDataset(tmp_path / "store", "vlm_image", SIZE, digest or encoder_digest("lerobot/pi05_base"))


def test_rows_are_shared_between_two_draws_of_the_same_layouts(tmp_path):
    a, b, stub = _layouts(0), _layouts(1, 3), TokenStub()
    store = _store(tmp_path)
    rows = store.rows_for(a, stub)
    assert (
        rows.dtype == np.int64 and rows.tolist() == list(range(5)) and stub.images == 5 and store.encoded == 5
    )
    assert (
        store.n_rows == 5
        and store.path == tmp_path / "store" / f"vlm_image_{SIZE}_{encoder_digest('lerobot/pi05_base')}"
    )
    again = _store(tmp_path)  # a new object on the same folder: nothing to encode, no encoder needed
    np.testing.assert_array_equal(again.rows_for(a, None), rows)
    assert again.encoded == 0
    mixed = Layouts.concat(
        [a[np.array([3, 1])], b, b[:1]]
    )  # two known layouts, three new ones (one of them twice)
    assert (
        again.rows_for(mixed, stub).tolist() == [3, 1, 5, 6, 7, 5] and again.encoded == 3 and stub.images == 8
    )
    meta = json.loads((again.path / "meta.json").read_text())
    assert {k: meta[k] for k in ("rows", "tokens", "width", "dtype", "source", "size")} == {
        "rows": 8,
        "tokens": 256,
        "width": 2048,
        "dtype": "float16",
        "source": "vlm_image",
        "size": SIZE,
    }
    assert meta["encoder_digest"] == encoder_digest("lerobot/pi05_base") and len(meta["encoder_digest"]) == 12
    assert (again.path / "tokens.f16").stat().st_size == 8 * 256 * 2048 * 2
    index = json.loads((again.path / "index.json").read_text())
    assert sorted(index.values()) == list(range(8)) and all(len(k) == 16 for k in index)


def test_a_copy_has_its_own_row_and_a_different_encoder_its_own_store(tmp_path):
    a = _layouts(0)
    moved = move_layouts(a, np.full(5, 0.3), np.zeros((5, 2)), np.zeros(5, bool))
    store = _store(tmp_path)
    rows, copies = store.rows_for(a, TokenStub()), store.rows_for(moved, TokenStub())
    assert not set(rows) & set(copies) and store.n_rows == 10
    assert layout_digests(a, SIZE) != layout_digests(a, SIZE + 16)  # the render spec is in the digest
    goal = Layouts(a.start, a.targets, a.mask, (a.goal + 1) % 4, a.obstacles, a.obstacle_mask)
    assert all(x != y for x, y in zip(layout_digests(a, SIZE), layout_digests(goal, SIZE), strict=True))
    assert encoder_digest("lerobot/pi05_base") != encoder_digest("elsewhere/pi05")
    other = _store(tmp_path, encoder_digest("elsewhere/pi05"))
    assert other.path != store.path and other.n_rows == 0


def test_gather_returns_the_stored_tokens_bit_for_bit_and_fills_a_pinned_buffer(tmp_path):
    a, stub = _layouts(0), TokenStub()
    store = _store(tmp_path)
    rows = store.rows_for(a, stub)
    tokens = torch.from_numpy(scene_for(a, "vlm_image", SIZE, TokenStub()))
    pick = rows[[4, 0, 0, 2]]
    got = store.gather(pick)
    assert (
        got.dtype == torch.float16 and got.shape == (4, 256, 2048) and torch.equal(got, tokens[[4, 0, 0, 2]])
    )
    buffer = torch.empty(6, 256, 2048, dtype=torch.float16, pin_memory=torch.cuda.is_available())
    into = store.gather(pick, buffer)
    assert into.data_ptr() == buffer.data_ptr() and torch.equal(into, tokens[[4, 0, 0, 2]])
    assert into.is_pinned() == torch.cuda.is_available()
    with pytest.raises(IndexError):
        store.gather(np.array([5]))
    sample = SceneDataset.sample_rows(rows[[0, 1, 2, 3, 4, 4]], 3, np.random.default_rng(0))
    assert len(sample) == 3 and set(sample) <= set(rows)
    np.testing.assert_array_equal(
        sample, SceneDataset.sample_rows(rows[[0, 1, 2, 3, 4, 4]], 3, np.random.default_rng(0))
    )
    assert (
        len(SceneDataset.sample_rows(rows, 2000, np.random.default_rng(0))) == 5
    )  # fewer rows than asked: all of them


def test_a_crashed_append_leaves_the_index_consistent(tmp_path, monkeypatch):
    a, b = _layouts(0), _layouts(1, 3)
    store = _store(tmp_path)
    store.rows_for(a, TokenStub())

    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(SceneDataset, "_write_index", crash)
    with pytest.raises(KeyboardInterrupt):  # the rows are written, the index is not
        store.rows_for(b, TokenStub())
    monkeypatch.undo()
    assert (store.path / "tokens.f16").stat().st_size == 8 * 256 * 2048 * 2
    reopened = _store(tmp_path)
    assert reopened.n_rows == 5 == len(json.loads((reopened.path / "index.json").read_text()))
    np.testing.assert_array_equal(reopened.rows_for(a, None), np.arange(5))
    assert reopened.rows_for(b, TokenStub()).tolist() == [5, 6, 7] and reopened.n_rows == 8
    assert (
        store.path / "tokens.f16"
    ).stat().st_size == 8 * 256 * 2048 * 2  # the orphan rows were overwritten
    want = torch.from_numpy(scene_for(Layouts.concat([a, b]), "vlm_image", SIZE, TokenStub()))
    assert torch.equal(reopened.gather(np.arange(8)), want)


def test_budgets_are_nested_at_one_seed(tmp_path):
    """Two time budgets at one seed draw nested demonstrations: the smaller budget's layouts are the larger's first
    rows, so a store filled by one budget serves those rows of the other."""
    args = train_adapter.parse_args(
        [
            "--kind",
            "task",
            "--obstacle",
            IMAGE,
            "--user-module",
            "u",
            "--corruption",
            "{rotation_deg: 60}",
            "--name",
            "x",
        ]
    )
    (small, *_), (large, *_) = (
        train_adapter.budget_demos(args, args.obstacle, np.random.default_rng(0), s, args.p_free)
        for s in (30.0, 60.0)
    )
    assert 0 < len(small) < len(large)
    assert layout_digests(small, SIZE) == layout_digests(large, SIZE)[: len(small)]
    store = _store(tmp_path)
    store.rows_for(small, TokenStub())
    store.rows_for(large, TokenStub())
    assert store.encoded == len(large) - len(small)
