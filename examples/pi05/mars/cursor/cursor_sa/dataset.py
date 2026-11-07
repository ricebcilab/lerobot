"""`SceneDataset`: a disk store of the frozen encoder's scene tokens keyed by layout, one row per distinct layout.

`SceneDataset(dir, source, size, encoder_digest)` lives in `<dir>/<source>_<size>_<encoder digest>/`:

- `tokens.f16`: a raw float16 array [rows, 256, 2048], grown by appending and read through `np.memmap`;
- `index.json`: {layout digest: row}, the rows the store holds (rows 0 .. n - 1, every one named once);
- `meta.json`: rows, tokens, width, dtype, source, size, encoder (the encoder's path when given), encoder_digest.

A layout's digest (`layout_digests`) is the sha256 of its start, targets, mask, goal, obstacles and obstacle mask (each
row's dtype, shape and bytes) and the render spec at `size`, 16 hex characters; the encoder's (`encoder_digest`) is the
sha256 of the encoder's path and prompt, 12 hex characters (the encoder part of the sweep's `scene_digest`). So a
layout drawn twice (two budgets of one seed, a split drawn again) is encoded once, and a moved copy, another render
size or another encoder gets rows of its own.

`rows_for(layouts, encoder)` returns every layout's row, rendering and encoding (`SceneEncoder.scene_array`, 256 layouts at a time) the
ones the store lacks and appending them. An append writes the rows, then the index (to a temporary file, renamed), then
the meta, under an exclusive lock on the folder, so writers in several processes take turns; a crash before the index is
renamed leaves the index naming only rows that were whole before, and the next append writes over the orphan bytes.
The encoding runs on forked torch generators, so whether the store already held a layout never changes a caller's draws.
`gather(rows, out)` copies rows into a CPU tensor (`out`, e.g. a pinned buffer, when given)."""

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from lerobot.envs.cursor import RENDER_MIN_RADIUS, RENDER_PALETTE, RENDER_SUPERSAMPLE, RENDER_TARGET_RADIUS
from lerobot.policies.steering.configuration_steering import VLM_SCENE_DIM, VLM_SCENE_TOKENS
from lerobot.policies.steering.processor_steering import SCENE_PROMPT, SceneEncoder

ENCODE_CHUNK = 256  # layouts rendered and encoded at a time
STORED_SOURCES = ("vlm_image", "vlm_prefix")  # the sources whose scene is one [256, 2048] token array per layout


def encoder_digest(vlm_path: str) -> str:
    """The first 12 hex digits of a sha256 over the encoder's path and the prompt `prefix_tokens` runs behind the
    image: the encoder part of the sweep's `scene_digest` spec."""
    spec = dict(vlm=str(vlm_path), prompt=SCENE_PROMPT)
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def layout_digests(layouts, size: int) -> list[str]:
    """Every layout's digest: the first 16 hex digits of a sha256 over its start, targets, mask, goal, obstacles and
    obstacle mask (each row's dtype, shape and bytes) and the render spec (`render_layouts` at `size`)."""
    spec = json.dumps(dict(size=int(size), palette=RENDER_PALETTE, supersample=RENDER_SUPERSAMPLE,
                           target_radius=RENDER_TARGET_RADIUS, min_radius=RENDER_MIN_RADIUS), sort_keys=True).encode()
    arrays = [np.ascontiguousarray(a) for a in (layouts.start, layouts.targets, layouts.mask, layouts.goal,
                                                 layouts.obstacles, layouts.obstacle_mask)]
    heads = [f"{a.dtype.str}{a.shape[1:]}".encode() for a in arrays]
    out = []
    for b in range(len(layouts)):
        h = hashlib.sha256()
        for head, a in zip(heads, arrays, strict=True):
            h.update(head + a[b].tobytes())
        h.update(spec)
        out.append(h.hexdigest()[:16])
    return out


def store_name(source: str, size: int, encoder_digest: str) -> str:
    """The folder a `SceneDataset` keeps its files in, under the store's directory."""
    return f"{source}_{int(size)}_{encoder_digest}"


class SceneDataset:
    """The scene tokens of `source` (vlm_image or vlm_prefix) rendered at `size` and encoded by the encoder of
    `encoder_digest`, one float16 [256, 2048] row per distinct layout (see the module docstring). `encoder_path` is
    recorded in the meta when given. The store holds no encoder: `rows_for` takes one, which may be None when
    every layout is stored."""

    def __init__(self, dir, source: str, size: int, encoder_digest: str, encoder_path: str | None = None):
        if source not in STORED_SOURCES:
            raise ValueError(f"a token store holds the sources {STORED_SOURCES}, not {source!r}")
        self.source, self.size, self.encoder_digest, self.encoder_path = source, int(size), encoder_digest, encoder_path
        self.tokens, self.width = VLM_SCENE_TOKENS, VLM_SCENE_DIM
        self.row_bytes = self.tokens * self.width * 2
        self._path = Path(dir) / store_name(source, self.size, encoder_digest)
        self._path.mkdir(parents=True, exist_ok=True)
        self.index = self._read_index()
        self.encoded = 0  # layouts the last rows_for call encoded
        self._memmap: np.memmap | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def n_rows(self) -> int:
        return len(self.index)

    def _read_index(self) -> dict[str, int]:
        file = self._path / "index.json"
        if not file.exists():
            return {}
        index = json.loads(file.read_text())
        if sorted(index.values()) != list(range(len(index))):
            raise ValueError(f"{file}: the index must name rows 0..{len(index) - 1} once each")
        size = (self._path / "tokens.f16").stat().st_size if (self._path / "tokens.f16").exists() else 0
        if size < len(index) * self.row_bytes:
            raise ValueError(f"{file} names {len(index)} rows but tokens.f16 holds {size // self.row_bytes}")
        return index

    @contextmanager
    def _locked(self):
        with open(self._path / "lock", "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _write_index(self, index: dict[str, int]) -> None:
        file = self._path / "index.json"
        tmp = file.with_name(f"index.json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(index))
        os.replace(tmp, file)

    def _write_meta(self) -> None:
        meta = dict(rows=self.n_rows, tokens=self.tokens, width=self.width, dtype="float16", source=self.source,
                    size=self.size, encoder=self.encoder_path, encoder_digest=self.encoder_digest)
        file = self._path / "meta.json"
        tmp = file.with_name(f"meta.json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, file)

    def _append(self, digests: list[str], tokens: np.ndarray) -> None:
        """Rows for `digests` (not stored when another writer stored them meanwhile), then the index, then the meta."""
        with self._locked():
            self.index = self._read_index()   # another process may have appended
            new = [j for j, d in enumerate(digests) if d not in self.index]
            if not new:
                return
            n = self.n_rows
            with open(self._path / "tokens.f16", "ab") as f:
                f.truncate(n * self.row_bytes)   # past the index: a crashed append's orphan rows
                f.write(np.ascontiguousarray(tokens[new], dtype=np.float16).tobytes())
                f.flush()
                os.fsync(f.fileno())
            index = dict(self.index)
            index.update({digests[j]: n + r for r, j in enumerate(new)})
            self._write_index(index)
            self.index = index
            self._write_meta()

    def rows_for(self, layouts, encoder) -> np.ndarray:
        """The row of every layout (int64 [B]), in the layouts' order; the layouts the store lacks are rendered and
        encoded (`SceneEncoder.scene_array` with `encoder`, ENCODE_CHUNK at a time, each distinct layout once) and appended in
        their first order. Sets `encoded` to the number of layouts this call encoded."""
        digests = layout_digests(layouts, self.size)
        self.index = self._read_index()
        missing = list(dict.fromkeys(d for d in digests if d not in self.index))
        self.encoded = 0
        if missing:
            first = {}
            for b, d in enumerate(digests):
                first.setdefault(d, b)
            at = np.array([first[d] for d in missing])
            # the encoder's lazy first load builds a model from torch's generators: a stored layout must not change a draw
            with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
                for lo in range(0, len(at), ENCODE_CHUNK):
                    chunk = at[lo:lo + ENCODE_CHUNK]
                    tokens = SceneEncoder.scene_array(layouts[chunk], self.source, self.size, encoder)
                    if tokens.shape[1:] != (self.tokens, self.width):
                        raise ValueError(f"the encoder gave tokens {tokens.shape[1:]}, not ({self.tokens}, {self.width})")
                    self._append([digests[b] for b in chunk], tokens)
                    self.encoded += len(chunk)
        return np.array([self.index[d] for d in digests], dtype=np.int64)

    def _rows_view(self) -> np.memmap:
        if self._memmap is None or len(self._memmap) < self.n_rows:
            self._memmap = np.memmap(self._path / "tokens.f16", dtype=np.float16, mode="r",
                                     shape=(self.n_rows, self.tokens, self.width))
        return self._memmap

    def gather(self, rows: np.ndarray, out: torch.Tensor | None = None) -> torch.Tensor:
        """The stored tokens of `rows`, float16 [len(rows), 256, 2048] on the CPU: written into the first len(rows)
        rows of `out` (e.g. a pinned buffer) when given, else into a new tensor."""
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) and (rows.min() < 0 or rows.max() >= self.n_rows):
            raise IndexError(f"rows must lie in 0..{self.n_rows - 1}")
        shape = (len(rows), self.tokens, self.width)
        if out is None:
            out = torch.empty(shape, dtype=torch.float16)
        elif out.dtype != torch.float16 or out.device.type != "cpu" or tuple(out.shape[1:]) != shape[1:] or len(out) < len(rows):
            raise ValueError(f"out must be a CPU float16 tensor [>= {len(rows)}, {self.tokens}, {self.width}]")
        out = out[:len(rows)]
        if len(rows):
            # mode="clip" (the rows are checked above): with "raise" numpy buffers the whole output first
            np.take(self._rows_view(), rows, axis=0, out=out.numpy(), mode="clip")
        return out

    @staticmethod
    def sample_rows(rows: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """min(n, len(rows)) of `rows` drawn without replacement with `rng`, in their order in `rows` (for the token
        reduction's fit)."""
        rows = np.asarray(rows)
        return rows[np.sort(rng.choice(len(rows), min(n, len(rows)), replace=False))]
