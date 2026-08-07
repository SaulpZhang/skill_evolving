"""Persistent task-embedding cache.

The cache is independent of a particular dataset or embedding provider.
Dataset adapters provide a namespace and the embedding configuration; the
text itself is hashed as the entry key. This lets train, validation, and
selection pools reuse identical task descriptions without assuming that
their task ids or traversal order are stable.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


_CACHE_VERSION = 1
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class EmbeddingCache:
    """A small, atomic ``.npz`` cache for text embeddings."""

    def __init__(
        self,
        cache_dir: str | os.PathLike | None = None,
        namespace: str = "default",
        config: dict[str, Any] | None = None,
        enabled: bool = True,
    ):
        root = Path(cache_dir) if cache_dir else _PROJECT_ROOT / "cache"
        if not root.is_absolute():
            root = _PROJECT_ROOT / root
        self.enabled = bool(enabled)
        self.namespace = str(namespace).strip().lower() or "default"
        self.config = {
            "version": _CACHE_VERSION,
            "namespace": self.namespace,
            **(config or {}),
        }
        fingerprint = hashlib.sha256(
            json.dumps(self.config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        self.path = root / self.namespace / f"embeddings_{fingerprint}.npz"
        self._vectors: dict[str, np.ndarray] = {}
        self._pending: dict[str, np.ndarray] = {}
        if self.enabled:
            self._load()

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            with np.load(self.path, allow_pickle=False) as archive:
                keys = archive["keys"]
                vectors = archive["embeddings"]
                if vectors.ndim != 2 or len(keys) != vectors.shape[0]:
                    raise ValueError("invalid embedding cache shape")
                self._vectors = {
                    str(key): np.asarray(vector, dtype=np.float32)
                    for key, vector in zip(keys.tolist(), vectors)
                }
        except Exception as exc:  # noqa: BLE001
            # A partial/corrupt cache should never prevent an experiment from
            # running; missing entries will simply be recomputed.
            print(f"  Warning: ignoring invalid embedding cache {self.path}: {exc}")
            self._vectors = {}

    def get(self, text: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        vector = self._vectors.get(self.key(text))
        return None if vector is None else vector.copy()

    def put(self, text: str, vector: Any) -> None:
        if not self.enabled:
            return
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise ValueError("Cannot cache an empty embedding")
        cache_key = self.key(text)
        self._vectors[cache_key] = array
        self._pending[cache_key] = array

    def save(self) -> bool:
        """Atomically persist new entries; return whether a file was written."""
        if not self.enabled or not self._pending:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Merge entries written by another process since this instance was
        # loaded. Atomic replacement prevents readers from seeing a partial
        # npz file when experiments share the project cache.
        latest: dict[str, np.ndarray] = {}
        if self.path.is_file():
            try:
                with np.load(self.path, allow_pickle=False) as archive:
                    keys = archive["keys"]
                    vectors = archive["embeddings"]
                    if vectors.ndim == 2 and len(keys) == vectors.shape[0]:
                        latest = {
                            str(key): np.asarray(vector, dtype=np.float32)
                            for key, vector in zip(keys.tolist(), vectors)
                        }
            except Exception:
                latest = {}
        latest.update(self._pending)
        keys = sorted(latest)
        vectors = np.stack([latest[key] for key in keys], axis=0)

        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.stem}.", suffix=".npz", dir=self.path.parent,
        )
        os.close(fd)
        try:
            np.savez_compressed(
                temporary,
                keys=np.asarray(keys, dtype="U64"),
                embeddings=vectors.astype(np.float32, copy=False),
            )
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self._vectors = latest
        self._pending.clear()
        return True

    @property
    def size(self) -> int:
        return len(self._vectors)
