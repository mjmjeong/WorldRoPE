"""Lossless 256x256 raw-image cache for the fixed RE10K training stream.

The cache changes only where decoded/resized RGB bytes are read from. Scene and
view selection, per-sample pose normalization, tensor conversion, and model
inputs remain on the legacy code path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from nvs.dataset import EvalDataset, _normalize_poses_identity_unit_distance


RAW_CACHE_FORMAT_VERSION = 1


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RE10KRaw256Cache:
    """Lazy mmap reader for lossless, pre-resized RE10K RGB arrays."""

    def __init__(
        self,
        cache_dir: os.PathLike[str] | str,
        *,
        expected_dataset_sha256: Optional[str] = None,
        expected_schedule_sha256: Optional[str] = None,
        require_complete: bool = True,
    ) -> None:
        self.cache_dir = os.path.realpath(os.fspath(cache_dir))
        metadata_path = Path(self.cache_dir) / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing RE10K raw-cache metadata: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        if int(self.metadata.get("format_version", -1)) != RAW_CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported RE10K raw-cache format: "
                f"{self.metadata.get('format_version')}"
            )
        if require_complete and not bool(self.metadata.get("build_complete", False)):
            raise ValueError(f"RE10K raw cache is incomplete: {self.cache_dir}")
        expected_contract = {
            "image_height": 256,
            "image_width": 256,
            "image_channels": 3,
            "image_dtype": "uint8",
            "image_storage": "lossless_raw_uint8_first_use_schedule_shards",
            "train_input_views": 2,
            "train_supervise_views": 1,
            "test_input_views": 2,
            "test_supervise_views": 3,
        }
        for key, value in expected_contract.items():
            if self.metadata.get(key) != value:
                raise ValueError(
                    f"RE10K raw-cache contract mismatch for {key}: "
                    f"{self.metadata.get(key)!r} != {value!r}"
                )
        if (
            expected_dataset_sha256 is not None
            and self.metadata.get("source_dataset_sha256")
            != expected_dataset_sha256
        ):
            raise ValueError("RE10K raw cache was built from a different dataset.")
        if (
            expected_schedule_sha256 is not None
            and self.metadata.get("train_schedule_sha256")
            != expected_schedule_sha256
        ):
            raise ValueError("RE10K raw cache was built for a different schedule.")

        self.images_per_shard = int(self.metadata["train_images_per_shard"])
        self.train_unique_images = int(self.metadata["train_unique_images"])
        self.test_scene_count = int(self.metadata["test_scene_count"])
        self._global_to_cache: Optional[np.ndarray] = None
        self._test_scene_names: Optional[np.ndarray] = None
        self._test_frame_ids: Optional[np.ndarray] = None
        self._test_relative_paths: Optional[np.ndarray] = None
        self._test_intrinsics: Optional[np.memmap] = None
        self._test_camtoworlds: Optional[np.memmap] = None
        self._test_images: Optional[np.memmap] = None
        self._test_scene_to_row: Optional[Dict[str, int]] = None
        self._train_image_mmaps: Dict[int, np.memmap] = {}
        self._train_intrinsic_mmaps: Dict[int, np.memmap] = {}

    def _ensure_train_index_open(self) -> None:
        if self._global_to_cache is None:
            self._global_to_cache = np.load(
                Path(self.cache_dir) / "train_global_to_cache.i32.npy",
                mmap_mode="r",
                allow_pickle=False,
            )

    def _train_shard(self, shard_index: int) -> Tuple[np.memmap, np.memmap]:
        shard_index = int(shard_index)
        images = self._train_image_mmaps.get(shard_index)
        intrinsics = self._train_intrinsic_mmaps.get(shard_index)
        if images is None or intrinsics is None:
            first = shard_index * self.images_per_shard
            count = min(self.images_per_shard, self.train_unique_images - first)
            if count <= 0:
                raise IndexError(f"Invalid RE10K cache shard: {shard_index}")
            image_path = (
                Path(self.cache_dir) / "train_shards" / f"images-{shard_index:06d}.u8"
            )
            intrinsic_path = (
                Path(self.cache_dir)
                / "train_shards"
                / f"intrinsics-{shard_index:06d}.f32"
            )
            images = np.memmap(
                image_path,
                mode="r",
                dtype=np.uint8,
                shape=(count, 256, 256, 3),
            )
            intrinsics = np.memmap(
                intrinsic_path,
                mode="r",
                dtype=np.float32,
                shape=(count, 3, 3),
            )
            self._train_image_mmaps[shard_index] = images
            self._train_intrinsic_mmaps[shard_index] = intrinsics
        return images, intrinsics

    def load_train(self, global_frame_index: int) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_train_index_open()
        assert self._global_to_cache is not None
        global_frame_index = int(global_frame_index)
        if not 0 <= global_frame_index < len(self._global_to_cache):
            raise IndexError(f"Invalid RE10K global frame index: {global_frame_index}")
        cache_index = int(self._global_to_cache[global_frame_index])
        if cache_index < 0:
            raise KeyError(
                f"Frame {global_frame_index} is absent from the fixed-schedule cache."
            )
        shard_index, local_index = divmod(cache_index, self.images_per_shard)
        images, intrinsics = self._train_shard(shard_index)
        return np.asarray(images[local_index]), np.asarray(intrinsics[local_index])

    def _ensure_test_open(self) -> None:
        if self._test_images is not None:
            return
        root = Path(self.cache_dir)
        self._test_scene_names = np.load(
            root / "test_scene_names.npy", mmap_mode="r", allow_pickle=False
        )
        self._test_frame_ids = np.load(
            root / "test_frame_ids.i32.npy", mmap_mode="r", allow_pickle=False
        )
        self._test_relative_paths = np.load(
            root / "test_relative_paths.npy", mmap_mode="r", allow_pickle=False
        )
        shape = (self.test_scene_count, 5)
        self._test_images = np.memmap(
            root / "test_images.u8",
            mode="r",
            dtype=np.uint8,
            shape=shape + (256, 256, 3),
        )
        self._test_intrinsics = np.memmap(
            root / "test_intrinsics.f32",
            mode="r",
            dtype=np.float32,
            shape=shape + (3, 3),
        )
        self._test_camtoworlds = np.memmap(
            root / "test_camtoworlds.f32",
            mode="r",
            dtype=np.float32,
            shape=shape + (4, 4),
        )
        assert self._test_scene_names is not None
        self._test_scene_to_row = {
            bytes(name).decode("utf-8"): row
            for row, name in enumerate(self._test_scene_names)
        }

    def test_row(self, scene_name: str) -> int:
        self._ensure_test_open()
        assert self._test_scene_to_row is not None
        try:
            return self._test_scene_to_row[scene_name]
        except KeyError as exc:
            raise KeyError(f"Scene {scene_name} is absent from the test cache.") from exc

    def load_test(
        self, row: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_test_open()
        assert self._test_images is not None
        assert self._test_intrinsics is not None
        assert self._test_camtoworlds is not None
        assert self._test_frame_ids is not None
        assert self._test_relative_paths is not None
        row = int(row)
        return (
            np.asarray(self._test_images[row]),
            np.asarray(self._test_intrinsics[row]),
            np.asarray(self._test_camtoworlds[row]),
            np.asarray(self._test_frame_ids[row]),
            np.asarray(self._test_relative_paths[row]),
        )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        for key in (
            "_global_to_cache",
            "_test_scene_names",
            "_test_frame_ids",
            "_test_relative_paths",
            "_test_intrinsics",
            "_test_camtoworlds",
            "_test_images",
            "_test_scene_to_row",
        ):
            state[key] = None
        state["_train_image_mmaps"] = {}
        state["_train_intrinsic_mmaps"] = {}
        return state


class RE10KRawCachedEvalDataset(EvalDataset):
    """The legacy EvalDataset ordering with cached, bit-identical payloads."""

    def __init__(self, *, image_cache_dir: str, **kwargs: Any) -> None:
        if int(kwargs.get("patch_size", 256)) != 256:
            raise ValueError("The RE10K raw cache requires dataset_patch_size=256.")
        if float(kwargs.get("zoom_factor", 1.0)) != 1.0:
            raise ValueError("The RE10K raw cache supports test zoom_factor=1.0 only.")
        if bool(kwargs.get("render_video", False)):
            raise ValueError("The fixed RE10K raw cache does not support render_video.")
        if int(kwargs.get("input_views", 2)) != 2:
            raise ValueError("The fixed RE10K raw cache requires two context views.")
        if int(kwargs.get("supervise_views", 3)) != 3:
            raise ValueError("The fixed RE10K raw cache requires three target views.")
        super().__init__(**kwargs)
        self.image_cache = RE10KRaw256Cache(image_cache_dir)

        index_name = kwargs.get("test_index_fp") or "evaluation_index_re10k.json"
        index_path = Path(index_name)
        if not index_path.is_absolute():
            index_path = Path(__file__).resolve().parent.parent / "assets" / index_name
        actual_index_sha = sha256_file(index_path)
        expected_index_sha = self.image_cache.metadata["test_index_sha256"]
        if actual_index_sha != expected_index_sha:
            raise ValueError(
                f"RE10K test-index hash mismatch: {actual_index_sha} != "
                f"{expected_index_sha}"
            )

    def __getitem__(self, scene_id: int) -> Dict[str, Any]:
        data_dir = bytes(self.data_dirs[scene_id]).decode("utf-8")
        scene_name = os.path.basename(data_dir)
        row = self.image_cache.test_row(scene_name)
        images, intrinsics, camtoworlds, cached_frame_ids, relative_paths = (
            self.image_cache.load_test(row)
        )
        requested_frame_ids = np.concatenate(
            [
                self.contexts[scene_id][: self.input_views],
                self.targets[scene_id][: self.supervise_views],
            ]
        ).astype(np.int32)
        if not np.array_equal(requested_frame_ids, cached_frame_ids):
            raise ValueError(
                f"Cached test views changed for {scene_name}: "
                f"{cached_frame_ids.tolist()} != {requested_frame_ids.tolist()}"
            )

        camtoworld = torch.from_numpy(np.array(camtoworlds, copy=True)).float()
        camtoworld = _normalize_poses_identity_unit_distance(
            camtoworld, ref0_idx=0, ref1_idx=self.input_views - 1
        )
        image_paths = [
            os.path.join(data_dir, bytes(path).decode("utf-8"))
            for path in relative_paths
        ]
        return {
            "camtoworld": camtoworld,
            "K": torch.from_numpy(np.array(intrinsics, copy=True)).float(),
            "image": torch.from_numpy(np.array(images, copy=True)).float(),
            "image_path": image_paths,
            "scene": scene_id,
        }
