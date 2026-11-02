"""Fast, deterministic Objaverse radial-0 datasets for URoPE.

The compact bundle stores camera metadata in NumPy mmap arrays and concatenates
the already-compressed JPEG byte streams into one blob per source shard.  No
image is re-encoded.  Training therefore performs no per-sample JSON parsing
and no per-image filesystem open/stat operation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np
import torch

from nvs.dataset import (
    _normalize_poses_identity_unit_distance,
    resize_crop_with_subpixel_accuracy,
)


FAST_OBJAVERSE_FORMAT_VERSION = 1
OPENCV_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ObjaverseFastDatasetBase(torch.utils.data.Dataset):
    """Shared mmap camera metadata and lazy JPEG-blob decoder."""

    def __init__(
        self,
        fast_dir: str,
        patch_size: int = 256,
        zoom_factor: float = 1.0,
        random_zoom: bool = False,
    ) -> None:
        self.fast_dir = os.path.realpath(fast_dir)
        self.patch_size = int(patch_size)
        self.zoom_factor = float(zoom_factor)
        self.random_zoom = bool(random_zoom)
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if self.random_zoom or self.zoom_factor != 1.0:
            raise ValueError(
                "The fixed Objaverse baseline disables random/explicit zoom; "
                "use random_zoom=false and zoom_factor=1.0."
            )

        manifest_path = Path(self.fast_dir) / "metadata.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing Objaverse fastdata manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        if int(self.metadata.get("format_version", -1)) != FAST_OBJAVERSE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported Objaverse fastdata format: "
                f"{self.metadata.get('format_version')}"
            )
        expected = {
            "radial_idx": 0,
            "views_per_object": 8,
            "image_width": 256,
            "image_height": 256,
            "pose_scale": "object_scale_ref0_identity_no_unit_baseline",
            "image_storage": "raw_jpeg_bytes_in_per_source_shard_blobs",
        }
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ValueError(
                    f"Objaverse fastdata contract mismatch for {key}: "
                    f"{self.metadata.get(key)!r} != {value!r}"
                )

        self.object_count = int(self.metadata["eligible_object_count"])
        self.image_width = int(self.metadata["image_width"])
        self.image_height = int(self.metadata["image_height"])
        self._object_uids: Optional[np.ndarray] = None
        self._shard_indices: Optional[np.ndarray] = None
        self._intrinsics: Optional[np.ndarray] = None
        self._camtoworlds: Optional[np.ndarray] = None
        self._jpeg_offsets: Optional[np.ndarray] = None
        self._shard_names: Optional[np.ndarray] = None
        self._blob_mmaps: Dict[int, np.memmap] = {}

    def _ensure_arrays_open(self) -> None:
        if self._intrinsics is not None:
            return
        root = Path(self.fast_dir)
        self._object_uids = np.load(
            root / "object_uids.npy", mmap_mode="r", allow_pickle=False
        )
        self._shard_indices = np.load(
            root / "shard_indices.npy", mmap_mode="r", allow_pickle=False
        )
        self._intrinsics = np.load(
            root / "intrinsics.f32.npy", mmap_mode="r", allow_pickle=False
        )
        self._camtoworlds = np.load(
            root / "camtoworlds_opencv.f32.npy", mmap_mode="r", allow_pickle=False
        )
        self._jpeg_offsets = np.load(
            root / "jpeg_offsets.u64.npy", mmap_mode="r", allow_pickle=False
        )
        self._shard_names = np.load(
            root / "shard_names.npy", mmap_mode="r", allow_pickle=False
        )
        n = self.object_count
        if self._object_uids.shape != (n,):
            raise ValueError(f"Invalid object_uids shape {self._object_uids.shape}")
        if self._shard_indices.shape != (n,):
            raise ValueError(f"Invalid shard_indices shape {self._shard_indices.shape}")
        if self._intrinsics.shape != (n, 8, 4):
            raise ValueError(f"Invalid intrinsics shape {self._intrinsics.shape}")
        if self._camtoworlds.shape != (n, 8, 4, 4):
            raise ValueError(f"Invalid camtoworlds shape {self._camtoworlds.shape}")
        if self._jpeg_offsets.shape != (n, 8, 2):
            raise ValueError(f"Invalid jpeg_offsets shape {self._jpeg_offsets.shape}")

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        for key in (
            "_object_uids",
            "_shard_indices",
            "_intrinsics",
            "_camtoworlds",
            "_jpeg_offsets",
            "_shard_names",
        ):
            state[key] = None
        state["_blob_mmaps"] = {}
        return state

    def _blob_for_shard(self, shard_index: int) -> np.memmap:
        self._ensure_arrays_open()
        assert self._shard_names is not None
        shard_index = int(shard_index)
        blob = self._blob_mmaps.get(shard_index)
        if blob is None:
            shard_name = bytes(self._shard_names[shard_index]).decode("utf-8")
            blob_path = Path(self.fast_dir) / "blobs" / f"{shard_name}.jpgblob"
            blob = np.memmap(blob_path, mode="r", dtype=np.uint8)
            self._blob_mmaps[shard_index] = blob
        return blob

    def object_uid(self, object_index: int) -> str:
        self._ensure_arrays_open()
        assert self._object_uids is not None
        return bytes(self._object_uids[int(object_index)]).decode("utf-8")

    def load_views(
        self, object_index: int, view_indices: Sequence[int]
    ) -> Dict[str, Any]:
        self._ensure_arrays_open()
        assert self._shard_indices is not None
        assert self._intrinsics is not None
        assert self._camtoworlds is not None
        assert self._jpeg_offsets is not None
        assert self._shard_names is not None

        object_index = int(object_index)
        if not 0 <= object_index < self.object_count:
            raise IndexError(f"Invalid Objaverse object index {object_index}")
        view_indices = [int(view) for view in view_indices]
        if any(view < 0 or view >= 8 for view in view_indices):
            raise IndexError(f"Invalid radial-0 view list: {view_indices}")

        shard_index = int(self._shard_indices[object_index])
        shard_name = bytes(self._shard_names[shard_index]).decode("utf-8")
        uid = self.object_uid(object_index)
        blob = self._blob_for_shard(shard_index)

        images = []
        Ks = []
        image_paths = []
        for view_index in view_indices:
            offset, length = self._jpeg_offsets[object_index, view_index]
            offset = int(offset)
            length = int(length)
            encoded = np.asarray(blob[offset : offset + length])
            image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(
                    f"Failed to decode packed JPEG for {shard_name}/{uid} "
                    f"view={view_index} offset={offset} length={length}"
                )
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            fx, fy, cx, cy = self._intrinsics[object_index, view_index]
            K = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            if image.shape[:2] != (self.patch_size, self.patch_size):
                image, K = resize_crop_with_subpixel_accuracy(
                    image, K, self.patch_size
                )
            images.append(image)
            Ks.append(K)
            image_paths.append(
                f"{self.metadata['source_root']}/{shard_name}/{uid}/views/"
                f"{view_index:03d}_0.jpg"
            )

        # The compact bundle stores OpenCV c2w matrices because that is the
        # convention used to validate the packed camera metadata.  Pristine
        # URoPE, however, deliberately expects Blender/OpenGL c2w matrices and
        # performs the Blender->OpenCV camera-axis conversion inside
        # ``urope._prepare_apply_fns``.  Convert the stored matrices back at
        # this loader boundary so that the internal URoPE conversion is applied
        # exactly once, matching ``ObjaverseTrainDataset``/``ObjaverseEvalDataset``.
        camtoworld_opencv = np.array(
            self._camtoworlds[object_index, view_indices], copy=True
        )
        camtoworld_blender = camtoworld_opencv @ OPENCV_TO_BLENDER
        camtoworld = torch.from_numpy(camtoworld_blender).float()
        camtoworld = _normalize_poses_identity_unit_distance(
            camtoworld, ref0_idx=0, ref1_idx=1
        )
        return {
            "camtoworld": camtoworld,
            "K": torch.from_numpy(np.stack(Ks)).float(),
            "image": torch.from_numpy(np.stack(images)).float(),
            "image_path": image_paths,
            "object_uid": uid,
        }


class ObjaverseFixedScheduleDataset(ObjaverseFastDatasetBase):
    """Two-rank, batch-eight, 80k fixed train stream."""

    def __init__(
        self,
        fast_dir: str,
        schedule_path: str,
        world_rank: int,
        max_steps: int,
        patch_size: int = 256,
        zoom_factor: float = 1.0,
        random_zoom: bool = False,
    ) -> None:
        super().__init__(fast_dir, patch_size, zoom_factor, random_zoom)
        self.schedule_path = os.path.realpath(schedule_path)
        self.world_rank = int(world_rank)
        self.max_steps = int(max_steps)
        if self.world_rank not in (0, 1):
            raise ValueError("The fixed Objaverse schedule requires exactly two ranks.")
        schedule_sha256 = _sha256_file(Path(self.schedule_path))
        expected_sha256 = self.metadata["train_schedule_sha256"]
        if schedule_sha256 != expected_sha256:
            raise ValueError(
                f"Objaverse schedule SHA-256 mismatch: "
                f"{schedule_sha256} != {expected_sha256}"
            )
        schedule = np.load(self.schedule_path, mmap_mode="r", allow_pickle=False)
        if schedule.dtype != np.int32 or schedule.shape != (2, 80_001, 8, 4):
            raise ValueError(
                f"Expected int32 schedule (2, 80001, 8, 4), got "
                f"{schedule.dtype} {schedule.shape}"
            )
        if self.max_steps != 80_000:
            raise ValueError("The fixed Objaverse v1 schedule requires max_steps=80000.")
        self._schedule: Optional[np.ndarray] = None

    def _ensure_schedule_open(self) -> None:
        if self._schedule is None:
            self._schedule = np.load(
                self.schedule_path, mmap_mode="r", allow_pickle=False
            )

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["_schedule"] = None
        return state

    def __len__(self) -> int:
        return (self.max_steps + 1) * 8

    def __getitem__(self, logical_index: int) -> Dict[str, Any]:
        self._ensure_schedule_open()
        assert self._schedule is not None
        logical_index = int(logical_index)
        step, slot = divmod(logical_index, 8)
        object_index, context0, context1, target = self._schedule[
            self.world_rank, step, slot
        ]
        return self.load_views(
            int(object_index), [int(context0), int(context1), int(target)]
        )


class ObjaverseFastEvalDataset(ObjaverseFastDatasetBase):
    """Fixed object-disjoint test set with one directed arc and 3 targets."""

    def __init__(
        self,
        fast_dir: str,
        patch_size: int = 256,
        zoom_factor: float = 1.0,
        first_n: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        input_views: int = 2,
        supervise_views: int = 3,
        render_video: bool = False,
        test_index_fp: Optional[str] = None,
    ) -> None:
        super().__init__(fast_dir, patch_size, zoom_factor, random_zoom=False)
        if render_video:
            raise NotImplementedError("Objaverse video rendering is not supported.")
        if test_index_fp is not None:
            raise ValueError("Objaverse fastdata always uses its immutable test index.")
        if input_views != 2 or supervise_views != 3:
            raise ValueError("Objaverse v1 evaluation requires 2 context and 3 targets.")
        root = Path(self.fast_dir)
        object_indices = np.load(
            root / "test_object_indices.i32.npy", allow_pickle=False
        )
        view_indices = np.load(root / "test_views.u8.npy", allow_pickle=False)
        if object_indices.shape != (6_474,) or view_indices.shape != (6_474, 5):
            raise ValueError(
                f"Invalid Objaverse test arrays: "
                f"{object_indices.shape}, {view_indices.shape}"
            )
        positions = np.arange(len(object_indices), dtype=np.int64)
        if first_n is not None:
            positions = positions[: int(first_n)]
        if rank is not None and world_size is not None:
            positions = positions[int(rank) :: int(world_size)]
        self._eval_object_indices = object_indices[positions].astype(np.int32)
        self._eval_view_indices = view_indices[positions].astype(np.uint8)

    def __len__(self) -> int:
        return len(self._eval_object_indices)

    def __getitem__(self, scene_index: int) -> Dict[str, Any]:
        scene_index = int(scene_index)
        loaded = self.load_views(
            int(self._eval_object_indices[scene_index]),
            self._eval_view_indices[scene_index].tolist(),
        )
        loaded["scene"] = scene_index
        return loaded
