"""Storage abstraction.

LocalStorageBackend is the required development backend.
S3StorageBackend can be added later behind the same interface without
touching callers. MinIO/S3 is NOT required for local development.
"""
from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO


class StorageBackend(ABC):
    """Minimal storage contract used by the card pipeline."""

    @abstractmethod
    def save_bytes(self, key: str, data: bytes) -> str:
        """Persist bytes under key. Returns the storage key."""

    @abstractmethod
    def save_file(self, key: str, src: BinaryIO) -> str:
        """Persist a readable binary stream under key."""

    @abstractmethod
    def open_path(self, key: str) -> Path:
        """Return a local filesystem path for the key (local backend only)."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...


class LocalStorageBackend(StorageBackend):
    """Filesystem-backed storage rooted at base_dir."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        path = (self.base_dir / key).resolve()
        if not str(path).startswith(str(self.base_dir.resolve())):
            raise ValueError(f"storage key escapes base dir: {key}")
        return path

    def save_bytes(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def save_file(self, key: str, src: BinaryIO) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return key

    def open_path(self, key: str) -> Path:
        return self._resolve(key)

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()
