"""Compact read-only mmap metadata backend for URoPE RE10K training.

The random scene/view selection and all image/pose preprocessing deliberately
match :class:`nvs.dataset.TrainDataset`.  Only repeated per-sample
``transforms.json`` parsing is replaced by shared memory-mapped arrays.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import imageio.v2 as imageio
import numpy as np
import torch

from nvs.dataset import (
    TrainDataset,
    _normalize_poses_identity_unit_distance,
    center_zoom_in_with_subpixel_accuracy,
    resize_crop_with_subpixel_accuracy,
)
from nvs.re10k_raw_cache import RE10KRaw256Cache


FAST_RE10K_FORMAT_VERSION = 1
LEGACY_UROPE_SCHEDULE_SHA256 = (
    "b156b22ff9a8b71c8055faff05d77f2867d31054c6b650a334d6c451a5a2a0d0"
)


class URoPEFastTrainDataset(TrainDataset):
    """URoPE vanilla sampler backed by compact mmap camera metadata."""

    def __init__(
        self,
        data_root: str,
        fast_dir: str,
        patch_size: int = 256,
        zoom_factor: float = 1.0,
        random_zoom: bool = False,
        input_views: int = 2,
        supervise_views: int = 1,
        image_cache_dir: Optional[str] = None,
    ) -> None:
        # Preserve the caller's path spelling in ``image_path`` so the fast
        # backend is also string-equivalent to the JSON loader.  Use realpath
        # only for validation against the bundle's source root.
        self.data_root = os.fspath(data_root).rstrip(os.sep)
        self.fast_dir = os.path.realpath(fast_dir)
        self.image_cache_dir = (
            None if image_cache_dir is None else os.path.realpath(image_cache_dir)
        )
        if random_zoom or float(zoom_factor) != 1.0:
            raise ValueError(
                "The compact RE10K backend requires random_zoom=false and "
                "zoom_factor=1.0 so it remains exactly equivalent to the "
                "validated URoPE training path."
            )
        if not os.path.isdir(os.path.abspath(self.data_root)):
            raise FileNotFoundError(f"Missing RE10K training root: {self.data_root}")

        manifest_path = Path(self.fast_dir) / "metadata.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing compact RE10K metadata: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        if int(self.metadata.get("format_version", -1)) != FAST_RE10K_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported compact RE10K format: "
                f"{self.metadata.get('format_version')}"
            )
        if self.metadata.get("canonical_image_pattern") != "images/{frame_id:05d}.png":
            raise ValueError("Compact metadata does not use canonical RE10K PNG paths.")

        scene_names = np.load(
            Path(self.fast_dir) / "scene_names.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if len(scene_names) != int(self.metadata["scene_count"]):
            raise ValueError("Compact scene count does not match scene_names.npy.")
        data_dirs = [
            os.path.join(self.data_root, bytes(scene_name).decode("utf-8"))
            for scene_name in scene_names
        ]
        # The clean-view dataset is a directory of scene symlinks into the
        # full-PNG root used to build the compact bundle.  Validate a few
        # sentinels without scanning/parsing all 66k JSON files at startup.
        built_root = os.path.realpath(self.metadata["data_root"])
        sentinel_indices = sorted({0, len(data_dirs) // 2, len(data_dirs) - 1})
        for scene_index in sentinel_indices:
            scene_name = bytes(scene_names[scene_index]).decode("utf-8")
            requested_scene = os.path.realpath(data_dirs[scene_index])
            built_scene = os.path.realpath(os.path.join(built_root, scene_name))
            if requested_scene != built_scene:
                raise ValueError(
                    "Compact metadata/data-root mismatch for scene "
                    f"{scene_name}: {requested_scene} != {built_scene}"
                )

        super().__init__(
            data_dirs=data_dirs,
            patch_size=patch_size,
            zoom_factor=zoom_factor,
            random_zoom=random_zoom,
            input_views=input_views,
            supervise_views=supervise_views,
        )
        if self.image_cache_dir is not None:
            if int(patch_size) != 256:
                raise ValueError(
                    "The lossless RE10K raw cache requires dataset_patch_size=256."
                )
            self.image_cache: Optional[RE10KRaw256Cache] = RE10KRaw256Cache(
                self.image_cache_dir,
                expected_dataset_sha256=self.metadata["dataset_sha256"],
                expected_schedule_sha256=LEGACY_UROPE_SCHEDULE_SHA256,
            )
        else:
            self.image_cache = None
        self._frame_offsets: Optional[np.ndarray] = None
        self._intrinsics: Optional[np.ndarray] = None
        self._source_frame_counts: Optional[np.ndarray] = None
        self._poses: Optional[np.memmap] = None

    def _ensure_arrays_open(self) -> None:
        if self._poses is not None:
            return
        root = Path(self.fast_dir)
        self._frame_offsets = np.load(
            root / "frame_offsets.npy", mmap_mode="r", allow_pickle=False
        )
        self._intrinsics = np.load(
            root / "intrinsics.npy", mmap_mode="r", allow_pickle=False
        )
        self._source_frame_counts = np.load(
            root / "source_frame_counts.npy", mmap_mode="r", allow_pickle=False
        )
        self._poses = np.memmap(
            root / "poses.f32",
            mode="r",
            dtype=np.float32,
            shape=(int(self.metadata["total_frames"]), 4, 4),
        )
        scene_count = int(self.metadata["scene_count"])
        if self._frame_offsets.shape != (scene_count + 1,):
            raise ValueError("Invalid compact frame_offsets shape.")
        if self._intrinsics.shape != (scene_count, 4):
            raise ValueError("Invalid compact intrinsics shape.")
        if self._source_frame_counts.shape != (scene_count,):
            raise ValueError("Invalid compact source_frame_counts shape.")

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        for key in (
            "_frame_offsets",
            "_intrinsics",
            "_source_frame_counts",
            "_poses",
        ):
            state[key] = None
        return state

    def load_preselected(
        self, scene_index: int, frame_ids: list[int]
    ) -> Dict[str, Any]:
        """Load a preselected legacy scene/view tuple without any RNG calls."""
        self._ensure_arrays_open()
        assert self._frame_offsets is not None
        assert self._intrinsics is not None
        assert self._source_frame_counts is not None
        assert self._poses is not None

        scene_index = int(scene_index)
        if not 0 <= scene_index < len(self.data_dirs):
            raise IndexError(f"Invalid compact RE10K scene index: {scene_index}")
        data_dir = bytes(self.data_dirs[scene_index]).decode("utf-8")

        frame_offset = int(self._frame_offsets[scene_index])
        frame_count = int(
            self._frame_offsets[scene_index + 1] - self._frame_offsets[scene_index]
        )
        frame_ids = [int(frame_id) for frame_id in frame_ids]
        if len(frame_ids) != self.input_views + self.supervise_views:
            raise ValueError(
                f"Expected {self.input_views + self.supervise_views} frame IDs, "
                f"got {len(frame_ids)}."
            )
        if any(frame_id < 0 or frame_id >= frame_count for frame_id in frame_ids):
            raise IndexError(
                f"Frame IDs {frame_ids} are outside scene {scene_index} with "
                f"{frame_count} frames."
            )

        fx, fy, cx, cy = self._intrinsics[scene_index]
        K_raw = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        images = []
        Ks = []
        c2ws = []
        image_paths = []
        for frame_id in frame_ids:
            image_path = os.path.join(data_dir, "images", f"{frame_id:05d}.png")
            if self.image_cache is None:
                image = imageio.imread(image_path)[..., :3]
                image, K = center_zoom_in_with_subpixel_accuracy(
                    image, K_raw, self.zoom_factor
                )
                image, K = resize_crop_with_subpixel_accuracy(
                    image, K, self.patch_size
                )
            else:
                image, K = self.image_cache.load_train(frame_offset + frame_id)
            c2w = np.asarray(self._poses[frame_offset + frame_id]) @ blender2opencv
            images.append(image)
            Ks.append(K)
            c2ws.append(c2w)
            image_paths.append(image_path)

        camtoworld = torch.from_numpy(np.stack(c2ws)).float()
        camtoworld = _normalize_poses_identity_unit_distance(
            camtoworld, ref0_idx=0, ref1_idx=self.input_views - 1
        )
        return {
            "camtoworld": camtoworld,
            "K": torch.from_numpy(np.stack(Ks)).float(),
            "image": torch.from_numpy(np.stack(images)).float(),
            "image_path": image_paths,
        }

    def __getitem__(self, _: Any) -> Dict[str, Any]:
        self._ensure_arrays_open()
        assert self._frame_offsets is not None

        # Retain the exact random calls of TrainDataset for JSON-vs-mmap audits.
        chosen_data_dir = np.random.choice(self.data_dirs)
        scene_index = int(np.searchsorted(self.data_dirs, chosen_data_dir))
        frame_count = int(
            self._frame_offsets[scene_index + 1] - self._frame_offsets[scene_index]
        )
        frame_ids = self._select_views(frame_count)
        if frame_ids is None:
            return self.__getitem__(None)
        frame_ids = sorted(frame_ids)
        frame_ids = [frame_ids[0], frame_ids[-1]] + frame_ids[1:-1]
        return self.load_preselected(scene_index, frame_ids)


class LegacyScheduleDataset(torch.utils.data.Dataset):
    """Read the exact legacy rank/step/slot selection from one mmap array."""

    def __init__(
        self,
        dataset: URoPEFastTrainDataset,
        schedule_path: str,
        world_rank: int,
        max_steps: int,
    ) -> None:
        self.dataset = dataset
        self.schedule_path = os.path.realpath(schedule_path)
        self.world_rank = int(world_rank)
        self.max_steps = int(max_steps)
        self._schedule: Optional[np.ndarray] = None
        digest = hashlib.sha256()
        with open(self.schedule_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        schedule_sha256 = digest.hexdigest()
        if schedule_sha256 != LEGACY_UROPE_SCHEDULE_SHA256:
            raise ValueError(
                "Legacy URoPE schedule hash mismatch: "
                f"{schedule_sha256} != {LEGACY_UROPE_SCHEDULE_SHA256}."
            )
        schedule = np.load(self.schedule_path, mmap_mode="r", allow_pickle=False)
        if (
            schedule.ndim != 4
            or schedule.shape[0] != 2
            or schedule.shape[1] < self.max_steps + 1
            or schedule.shape[2:] != (8, 4)
        ):
            raise ValueError(
                f"Legacy schedule shape {schedule.shape} does not match "
                f"(2, >= {self.max_steps + 1}, 8, 4)."
            )
        if schedule.dtype != np.int32:
            raise ValueError(f"Legacy schedule must be int32, got {schedule.dtype}.")
        if self.world_rank not in (0, 1):
            raise ValueError("The legacy schedule is defined for exactly two ranks.")
        if dataset.input_views != 2 or dataset.supervise_views != 1:
            raise ValueError(
                "The recovered legacy schedule requires input_views=2 and "
                "supervise_views=1."
            )
        del schedule

    def _ensure_schedule_open(self) -> None:
        if self._schedule is None:
            self._schedule = np.load(
                self.schedule_path, mmap_mode="r", allow_pickle=False
            )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_schedule"] = None
        return state

    def __len__(self) -> int:
        return (self.max_steps + 1) * 8

    def __getitem__(self, logical_index: int) -> Dict[str, Any]:
        self._ensure_schedule_open()
        assert self._schedule is not None
        step, slot = divmod(int(logical_index), 8)
        if not 0 <= step <= self.max_steps:
            raise IndexError(f"Legacy schedule step outside range: {step}")
        record = self._schedule[self.world_rank, step, slot]
        scene_index = int(record[0])
        if scene_index < 0:
            raise IndexError(
                f"Legacy schedule padding accessed at step={step}, slot={slot}."
            )
        return self.dataset.load_preselected(scene_index, record[1:].tolist())
