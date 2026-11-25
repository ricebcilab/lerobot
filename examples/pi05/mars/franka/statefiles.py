"""A session's chunk-boundary states as one npz: the file the user-module trainer and validator read.

Phase B trains the user module on states a session actually visited, never on human labels
(DESIGN.md, "Phase B"). At each chunk boundary the loop already holds everything such a row needs: the
raw observation the policy's chunk starts from (the model-ready images and the arm's state), the
operator's deflection, the *served* command the method would steer, and the nominal pose ``q0`` the
corruption lift was taken at. This module freezes exactly that row, once per chunk boundary, in a plain
``numpy.savez`` archive -- no pickle, so a file can be read without any of the session's objects, its
config or the station.

The format is the frozen one from DESIGN.md's "Phase B" (``format_version`` 1):

====================  ======================  =========================================================
key                   shape, dtype            what it is
====================  ======================  =========================================================
image_exterior        (N, 3, 224, 224) uint8  the boundary's exterior image, model-ready letterbox
image_wrist           (N, 3, 224, 224) uint8  the boundary's wrist image, same convention
joint_position        (N, 7) float64          the arm's joint angles before the chunk's first action
gripper               (N, 1) float64          the station's gripper position before that action
deflection            (N, 6) float64          the operator's deflection the boundary's command came from
served                (N, 7) float64          the served command in force for that chunk, action units
q0                    (7,) float64            the session's nominal pose (the corruption lift's ``q0``)
episode               (N,) int32              the episode the tick belonged to, 0-based (the
                                              index a ``{random: ...}`` readout drew F from)
tick                  (N,) int32              the loop tick inside that episode, 0-based
prompt                (N,) unicode            the task text the batch carried
mode                  (N,) unicode            the loop's mode: ``teleop``, ``autonomous`` or ``steered``
meta                  () unicode              the session's provenance, a 0-d array of a JSON object
====================  ======================  =========================================================

**Images are stored as uint8.** The batch holds ``float32`` in [0, 1] because the checkpoint's
``img*2-1`` would destroy a 0-255 array (DESIGN.md, "What the checkpoint's own code settles"). The file
stores ``round(255*x)`` instead: 300 kB per state rather than 1.2 MB, which is what makes a few hundred
states a reasonable thing to collect. The round trip is exact for images that came from a station
frame, because the station's chain is ``uint8 frame -> float32(frame)/255 -> model`` and
``round(255*float32(k/255)) == k``; only a batch image that never was a uint8 frame can be off by one.

**``q0`` arrives as an ``add`` argument**, not in ``meta``. It is the loop's to resolve (``command.q0``
when configured, else the station's reset pose captured at the first episode), and the CLI cannot know
a station's reset pose before the run; the recorder is built before the loop runs, so only the loop can
supply it. It is a session constant: a second ``add`` with a different ``q0`` is refused.

**``meta``** is free-form provenance; the CLI writes the config path and its SHA-256, the policy and
statistics paths, the station's host/arm/rate, the readout (corruption spec, noise, seed) and an ISO
timestamp. The recorder owns ``format_version`` and stamps the one it writes; a caller may not write
another. ``load`` refuses a file of another version rather than guessing what its arrays mean.

Pure numpy: importing this pulls neither torch nor c3po, so it stays on ``--dry-run``'s path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from bridge import ARM_DIMS, TWIST_DIMS
from numpy.typing import NDArray
from station import IMAGE_SIZE

FORMAT_VERSION = 1
"""The only state-file format this module writes or reads; ``load`` names it in its refusals."""

_IMAGE_FEEDS = {
    "image_exterior": "observation.images.base_0_rgb",
    "image_wrist": "observation.images.left_wrist_0_rgb",
}
"""The batch keys the two stored images come from (``station.observation_batch``'s roles)."""

_ROW_TAILS: dict[str, tuple[int, ...]] = {
    "image_exterior": (3, IMAGE_SIZE, IMAGE_SIZE),
    "image_wrist": (3, IMAGE_SIZE, IMAGE_SIZE),
    "joint_position": (ARM_DIMS,),
    "gripper": (1,),
    "deflection": (TWIST_DIMS,),
    "served": (ARM_DIMS,),
    "episode": (),
    "tick": (),
    "prompt": (),
    "mode": (),
}
"""Every per-state array's shape after the leading ``N``; the format's row columns."""

_ROW_DTYPES: dict[str, np.dtype | str] = {
    "image_exterior": np.dtype(np.uint8),
    "image_wrist": np.dtype(np.uint8),
    "joint_position": np.dtype(np.float64),
    "gripper": np.dtype(np.float64),
    "deflection": np.dtype(np.float64),
    "served": np.dtype(np.float64),
    "episode": np.dtype(np.int32),
    "tick": np.dtype(np.int32),
    "prompt": "U",  # any unicode width: a file's prompts are one dtype
    "mode": "U",
}
"""Every per-state array's dtype; ``U`` stands for any unicode width."""


# ---------------------------------------------------------------- writing


class Recorder:
    """One session's chunk-boundary states, buffered in memory until ``save``.

    ``meta`` is the session's provenance mapping (JSON-serializable); it is checked here so a bad
    value fails at construction rather than after a run. ``add`` appends one boundary row; the loop
    calls it once per chunk in every mode, with the *pre-action* observation, and ``save`` writes the
    whole buffer as one ``np.savez`` file. ``save`` is the caller's job (the CLI's, like the dataset
    download): a recorder never writes on its own, so a run killed mid-episode leaves no partial file.

    The buffer is one 300 kB image pair per state; a few hundred states is a few tens of MB of RAM,
    which is the reason the format stores images as uint8.
    """

    def __init__(self, meta: Mapping[str, object]) -> None:
        caller_version = meta.get("format_version")
        if caller_version is not None and caller_version != FORMAT_VERSION:
            raise ValueError(
                f"meta.format_version: this recorder writes format {FORMAT_VERSION}, not {caller_version!r}"
            )
        self._meta: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            **{key: value for key, value in meta.items() if key != "format_version"},
        }
        try:
            json.dumps(self._meta, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"meta: the provenance must be JSON-serializable ({error})") from error
        self._columns: dict[str, list[Any]] = {name: [] for name in _ROW_TAILS}
        self._images: dict[str, list[NDArray]] = {name: [] for name in _IMAGE_FEEDS}
        self._q0: NDArray | None = None

    def add(
        self,
        batch: Mapping,
        *,
        joint_position: NDArray,
        gripper: NDArray,
        deflection: NDArray,
        served: NDArray,
        q0: NDArray,
        episode: int,
        tick: int,
        prompt: str,
        mode: str,
    ) -> None:
        """Append one chunk-boundary row: the state before that chunk's first action.

        ``batch`` is the loop's raw batch (``station.observation_batch``'s dict, before the
        pre-processor): its two image keys become the stored uint8 images. ``joint_position`` (7,),
        ``gripper`` (1,), ``deflection`` (6,) and ``served`` (7,) are the values in force at that
        boundary; ``q0`` (7,) is the session's nominal pose (see the module docstring for why it
        travels here); ``episode``/``tick`` are the loop's 0-based indices, ``prompt`` and ``mode``
        the batch's task text and the loop's mode. A shape or dtype that does not fit is refused by
        name, before it can reach a file.
        """
        if not isinstance(batch, Mapping):
            raise ValueError(f"batch: expected a mapping (a raw observation batch), got {type(batch)}")
        for name, key in _IMAGE_FEEDS.items():
            if key not in batch:
                raise ValueError(
                    f"batch: no {key!r} for {name!r}; the batch must come from station.observation_batch"
                )
        images = {name: _image(batch[key], name) for name, key in _IMAGE_FEEDS.items()}
        row = {
            "joint_position": _vector(joint_position, ARM_DIMS, "joint_position"),
            "gripper": _vector(gripper, 1, "gripper"),
            "deflection": _vector(deflection, TWIST_DIMS, "deflection"),
            "served": _vector(served, ARM_DIMS, "served"),
            "episode": _index(episode, "episode"),
            "tick": _index(tick, "tick"),
            "prompt": _text(prompt, "prompt"),
            "mode": _text(mode, "mode"),
        }
        pose = _pose(q0)
        if self._q0 is not None and not np.array_equal(self._q0, pose):
            raise ValueError(
                "q0: a states file is one session's nominal pose; the loop re-resolved it mid-recording"
            )
        # Everything validated: append atomically, so a refused row does not leave a partial one.
        self._q0 = pose
        for name, image in images.items():
            self._images[name].append(image)
        for name, value in row.items():
            self._columns[name].append(value)

    def __len__(self) -> int:
        return len(self._columns["episode"])

    def save(self, path: Path) -> Path:
        """Write the buffered states to ``path`` (its parent created) and return ``path``.

        ``numpy.savez`` with a file object, so the file is written exactly at ``path`` (``np.savez``
        on a string appends ``.npz`` when it is missing; a caller that wants the suffix writes it).
        The file holds no pickled object: every array is plain numeric or unicode. An empty recorder
        writes N = 0 arrays and ``q0`` of zeros, because no state supplied a pose to store.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self._arrays()
        with open(path, "wb") as file:
            np.savez(file, **arrays)
        return path

    def _arrays(self) -> dict[str, NDArray]:
        """The recorder's buffer as the format's arrays, in the format's names."""
        arrays: dict[str, NDArray] = {}
        for name, tail in _ROW_TAILS.items():
            rows = self._images[name] if name in _IMAGE_FEEDS else self._columns[name]
            if tail:
                arrays[name] = _stack(rows, tail, _ROW_DTYPES[name])
            elif _ROW_DTYPES[name] == "U":
                arrays[name] = np.asarray(rows, dtype=np.str_)
            else:
                arrays[name] = np.asarray(rows, dtype=_ROW_DTYPES[name])
        arrays["q0"] = np.zeros(ARM_DIMS) if self._q0 is None else self._q0
        arrays["meta"] = np.array(json.dumps(self._meta, sort_keys=True), dtype=np.str_)
        return arrays


# ---------------------------------------------------------------- reading


def load(path: Path) -> tuple[dict[str, NDArray], dict]:
    """Read a states file: its arrays and its ``meta`` (the JSON object), both from the archive.

    Refuses a file that is not this format, naming what is wrong: a missing or non-JSON ``meta``, a
    ``format_version`` other than :data:`FORMAT_VERSION`, a missing or unknown array, and any array
    whose shape or dtype is not the format's. The archive is opened with ``allow_pickle=False``, so a
    file that carries a pickled object cannot execute one.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as file:
        arrays = {name: np.array(file[name], copy=True) for name in file.files}
    meta = _meta(arrays.pop("meta", None), path)
    _check_arrays(arrays, path)
    return arrays, meta


# ---------------------------------------------------------------- validated pieces


def _meta(array: NDArray | None, path: Path) -> dict:
    """The archive's ``meta`` array as the JSON object it holds, or a named refusal."""
    if array is None:
        raise ValueError(f"{path}: no 'meta' array; this is not a format-{FORMAT_VERSION} states file")
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(
            f"{path}: meta: expected a 0-d unicode array of JSON, got shape {array.shape} dtype {array.dtype}"
        )
    try:
        meta = json.loads(str(array[()]))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: meta: expected a JSON object, {error}") from error
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: meta: expected a JSON object, got {type(meta).__name__}")
    version = meta.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: meta.format_version: expected {FORMAT_VERSION}, got {version!r}; "
            f"this loader reads only format {FORMAT_VERSION}"
        )
    return meta


def _check_arrays(arrays: dict[str, NDArray], path: Path) -> None:
    """Every array the format names, with its shape and dtype; nothing else, and all N rows."""
    expected = {*_ROW_TAILS, "q0"}
    unknown = sorted(set(arrays) - expected)
    if unknown:
        raise ValueError(
            f"{path}: unknown array {unknown[0]!r} (format {FORMAT_VERSION} has {sorted(expected)})"
        )
    missing = sorted(expected - set(arrays))
    if missing:
        raise ValueError(
            f"{path}: missing array {missing[0]!r} (format {FORMAT_VERSION} needs {sorted(expected)})"
        )
    rows = arrays["joint_position"].shape[0] if arrays["joint_position"].ndim else 0
    for name, tail in _ROW_TAILS.items():
        array = arrays[name]
        if array.ndim != 1 + len(tail) or array.shape[1:] != tail or array.shape[0] != rows:
            raise ValueError(
                f"{path}: {name}: expected shape ({rows}, {_shape_text(tail)}) -- one row per "
                f"joint_position row -- got {array.shape}"
            )
        _check_dtype(array, _ROW_DTYPES[name], name, path)
    q0 = arrays["q0"]
    if q0.shape != (ARM_DIMS,):
        raise ValueError(f"{path}: q0: expected shape ({ARM_DIMS},), got {q0.shape}")
    _check_dtype(q0, np.dtype(np.float64), "q0", path)


def _check_dtype(array: NDArray, expected: np.dtype | str, name: str, path: Path) -> None:
    if expected == "U":
        if array.dtype.kind != "U":
            raise ValueError(f"{path}: {name}: expected unicode dtype, got {array.dtype}")
        return
    if array.dtype != expected:
        raise ValueError(f"{path}: {name}: expected dtype {np.dtype(expected)}, got {array.dtype}")


def _stack(rows: list[NDArray], tail: tuple[int, ...], dtype: np.dtype) -> NDArray:
    """The buffered rows as one ``(N, *tail)`` array; an empty buffer is ``(0, *tail)``."""
    if not rows:
        return np.empty((0, *tail), dtype=dtype)
    return np.stack(rows).astype(dtype, copy=False)


def _shape_text(tail: tuple[int, ...]) -> str:
    """A tail shape as it reads in an error message: ``3, 224, 224``, or ``nothing`` for a scalar."""
    return ", ".join(str(value) for value in tail) if tail else "nothing"


def _image(value: NDArray, name: str) -> NDArray:
    """One batch image as the file's uint8: ``float32 (3, 224, 224)`` in [0, 1] -> ``round(255*x)``."""
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"{name}: expected a batch image of shape (3, {IMAGE_SIZE}, {IMAGE_SIZE}), got {array.shape}"
        )
    if not np.all(np.isfinite(array)) or array.min() < -1e-6 or array.max() > 1.0 + 1e-6:
        raise ValueError(
            f"{name}: expected a batch image in [0, 1], got range [{array.min()}, {array.max()}]"
        )
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _vector(value: NDArray, size: int, name: str) -> NDArray:
    """One state vector as float64 ``(size,)``, finite: the buffer's numeric row type."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name}: expected shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: expected finite numbers, got {array}")
    return array


def _pose(value: NDArray) -> NDArray:
    """``q0`` as float64 ``(7,)``: one session's nominal pose, a constant of the recording."""
    return _vector(value, ARM_DIMS, "q0")


def _index(value: int, name: str) -> int:
    """A 0-based loop index as a Python int (the format's int32 columns)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name}: expected a non-negative integer, got {value!r}")
    return value


def _text(value: str, name: str) -> str:
    """A unicode column value (the task text or the loop's mode); the empty string is legal."""
    if not isinstance(value, str):
        raise ValueError(f"{name}: expected a string, got {type(value).__name__}")
    return value
