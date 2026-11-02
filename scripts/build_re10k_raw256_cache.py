#!/usr/bin/env python3
"""Build a resumable, lossless raw-uint8 cache for the fixed RE10K stream."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np


FORMAT_VERSION = 1
EXPECTED_SCHEDULE_SHA256 = (
    "b156b22ff9a8b71c8055faff05d77f2867d31054c6b650a334d6c451a5a2a0d0"
)
EXPECTED_TEST_INDEX_SHA256 = (
    "fb695e3aa97bacc61e9d6530cadbe200f410ef93a82773b11b6bd76c736756c5"
)
IMAGE_BYTES = 256 * 256 * 3
BLENDER_TO_OPENCV = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)

_WORKER_DATA_ROOT: Path
_WORKER_FRAME_OFFSETS: np.ndarray
_WORKER_INTRINSICS: np.ndarray
_WORKER_SCENE_NAMES: np.ndarray


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def resize_crop_exact(
    image: np.ndarray, intrinsic: np.ndarray, patch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """Byte-for-byte copy of nvs.dataset.resize_crop_with_subpixel_accuracy."""
    h, w = image.shape[:2]
    scale = patch_size / min(h, w)
    new_w, new_h = w * scale, h * scale
    crop_x = (new_w - patch_size) / 2
    crop_y = (new_h - patch_size) / 2
    transform = np.array(
        [[scale, 0, -crop_x], [0, scale, -crop_y]], dtype=np.float32
    )
    interpolation = cv2.INTER_AREA if min(h, w) > patch_size else cv2.INTER_CUBIC
    resized = cv2.warpAffine(
        image, transform, (patch_size, patch_size), flags=interpolation
    )
    resized_intrinsic = intrinsic.copy()
    resized_intrinsic[:2, :] *= scale
    resized_intrinsic[0, 2] -= crop_x
    resized_intrinsic[1, 2] -= crop_y
    return resized, resized_intrinsic


def _decode_resize(path: Path, intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    image = imageio.imread(path)[..., :3]
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unexpected image payload {path}: {image.dtype} {image.shape}")
    image, intrinsic = resize_crop_exact(image, intrinsic, 256)
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise ValueError(f"Unexpected resized image {path}: {image.dtype} {image.shape}")
    return np.ascontiguousarray(image), np.ascontiguousarray(intrinsic, dtype=np.float32)


def _init_train_worker(data_root: str, fast_dir: str) -> None:
    global _WORKER_DATA_ROOT
    global _WORKER_FRAME_OFFSETS
    global _WORKER_INTRINSICS
    global _WORKER_SCENE_NAMES
    _WORKER_DATA_ROOT = Path(data_root)
    fast = Path(fast_dir)
    _WORKER_FRAME_OFFSETS = np.load(
        fast / "frame_offsets.npy", mmap_mode="r", allow_pickle=False
    )
    _WORKER_INTRINSICS = np.load(
        fast / "intrinsics.npy", mmap_mode="r", allow_pickle=False
    )
    _WORKER_SCENE_NAMES = np.load(
        fast / "scene_names.npy", mmap_mode="r", allow_pickle=False
    )


def _build_train_image(global_frame_index: int) -> Tuple[np.ndarray, np.ndarray]:
    global_frame_index = int(global_frame_index)
    scene_index = int(
        np.searchsorted(_WORKER_FRAME_OFFSETS, global_frame_index, side="right") - 1
    )
    frame_index = global_frame_index - int(_WORKER_FRAME_OFFSETS[scene_index])
    scene_name = bytes(_WORKER_SCENE_NAMES[scene_index]).decode("utf-8")
    fx, fy, cx, cy = _WORKER_INTRINSICS[scene_index]
    intrinsic = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    path = _WORKER_DATA_ROOT / scene_name / "images" / f"{frame_index:05d}.png"
    return _decode_resize(path, intrinsic)


def _build_test_scene(
    item: Tuple[str, Sequence[int], str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    scene_name, frame_ids, test_root = item
    data_dir = Path(test_root) / scene_name
    with (data_dir / "transforms.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    intrinsic_raw = np.array(
        [
            [metadata["fl_x"], 0.0, metadata["cx"]],
            [0.0, metadata["fl_y"], metadata["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    images: List[np.ndarray] = []
    intrinsics: List[np.ndarray] = []
    camtoworlds: List[np.ndarray] = []
    relative_paths: List[str] = []
    frames = metadata["frames"]
    for frame_id in frame_ids:
        frame = frames[int(frame_id)]
        relative_path = frame["file_path"]
        image, intrinsic = _decode_resize(data_dir / relative_path, intrinsic_raw)
        camtoworld = (
            np.array(frame["transform_matrix"], dtype=np.float32)
            @ BLENDER_TO_OPENCV
        )
        images.append(image)
        intrinsics.append(intrinsic)
        camtoworlds.append(camtoworld)
        relative_paths.append(relative_path)
    return (
        np.stack(images),
        np.stack(intrinsics).astype(np.float32, copy=False),
        np.stack(camtoworlds).astype(np.float32, copy=False),
        relative_paths,
    )


def build_train_plan(
    output_dir: Path, fast_dir: Path, schedule_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    mapping_path = output_dir / "train_global_to_cache.i32.npy"
    globals_path = output_dir / "train_cache_global_indices.i32.npy"
    if mapping_path.is_file() and globals_path.is_file():
        mapping = np.load(mapping_path, mmap_mode="r", allow_pickle=False)
        globals_in_cache = np.load(globals_path, mmap_mode="r", allow_pickle=False)
        return mapping, globals_in_cache

    schedule = np.load(schedule_path, mmap_mode="r", allow_pickle=False)
    frame_offsets = np.load(
        fast_dir / "frame_offsets.npy", mmap_mode="r", allow_pickle=False
    )
    total_frames = int(frame_offsets[-1])
    mapping = np.full(total_frames, -1, dtype=np.int32)
    globals_in_cache: List[int] = []
    # Interleave ranks by step. This preserves forward locality for both ranks.
    for step in range(schedule.shape[1]):
        for rank in range(schedule.shape[0]):
            for slot in range(schedule.shape[2]):
                scene, context0, context1, target = map(
                    int, schedule[rank, step, slot]
                )
                if scene < 0:
                    continue
                frame_count = int(frame_offsets[scene + 1] - frame_offsets[scene])
                for frame in (context0, context1, target):
                    if not 0 <= frame < frame_count:
                        raise ValueError(
                            f"Invalid schedule frame: {(rank, step, slot, scene, frame)}"
                        )
                    global_frame = int(frame_offsets[scene]) + frame
                    if mapping[global_frame] < 0:
                        mapping[global_frame] = len(globals_in_cache)
                        globals_in_cache.append(global_frame)
    globals_array = np.asarray(globals_in_cache, dtype=np.int32)
    atomic_npy(mapping_path, mapping)
    atomic_npy(globals_path, globals_array)
    return (
        np.load(mapping_path, mmap_mode="r", allow_pickle=False),
        np.load(globals_path, mmap_mode="r", allow_pickle=False),
    )


def build_train_shards(
    *,
    output_dir: Path,
    data_root: Path,
    fast_dir: Path,
    globals_in_cache: np.ndarray,
    images_per_shard: int,
    workers: int,
    limit: int | None,
) -> Dict[str, Dict[str, str]]:
    shard_dir = output_dir / "train_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    count = len(globals_in_cache) if limit is None else min(limit, len(globals_in_cache))
    shard_count = math.ceil(count / images_per_shard)
    checksums_path = output_dir / "train_shard_checksums.json"
    if checksums_path.is_file():
        with checksums_path.open("r", encoding="utf-8") as handle:
            checksums: Dict[str, Dict[str, str]] = json.load(handle)
    else:
        checksums = {}
    started = time.monotonic()
    newly_built = 0
    context = mp.get_context("fork")
    with context.Pool(
        processes=workers,
        initializer=_init_train_worker,
        initargs=(os.fspath(data_root), os.fspath(fast_dir)),
        maxtasksperchild=4096,
    ) as pool:
        for shard_index in range(shard_count):
            first = shard_index * images_per_shard
            last = min(first + images_per_shard, count)
            shard_size = last - first
            image_path = shard_dir / f"images-{shard_index:06d}.u8"
            intrinsic_path = shard_dir / f"intrinsics-{shard_index:06d}.f32"
            expected_image_bytes = shard_size * IMAGE_BYTES
            expected_intrinsic_bytes = shard_size * 3 * 3 * 4
            key = f"{shard_index:06d}"
            if (
                image_path.is_file()
                and intrinsic_path.is_file()
                and image_path.stat().st_size == expected_image_bytes
                and intrinsic_path.stat().st_size == expected_intrinsic_bytes
                and key in checksums
            ):
                continue

            image_partial = image_path.with_suffix(image_path.suffix + ".partial")
            intrinsic_partial = intrinsic_path.with_suffix(
                intrinsic_path.suffix + ".partial"
            )
            image_partial.unlink(missing_ok=True)
            intrinsic_partial.unlink(missing_ok=True)
            image_digest = hashlib.sha256()
            intrinsic_digest = hashlib.sha256()
            ids: Iterable[int] = map(int, globals_in_cache[first:last])
            with image_partial.open("wb", buffering=16 * 1024 * 1024) as image_handle:
                with intrinsic_partial.open(
                    "wb", buffering=1024 * 1024
                ) as intrinsic_handle:
                    for image, intrinsic in pool.imap(
                        _build_train_image, ids, chunksize=4
                    ):
                        image_bytes = image.tobytes(order="C")
                        intrinsic_bytes = intrinsic.tobytes(order="C")
                        image_handle.write(image_bytes)
                        intrinsic_handle.write(intrinsic_bytes)
                        image_digest.update(image_bytes)
                        intrinsic_digest.update(intrinsic_bytes)
                    image_handle.flush()
                    intrinsic_handle.flush()
                    os.fsync(image_handle.fileno())
                    os.fsync(intrinsic_handle.fileno())
            if image_partial.stat().st_size != expected_image_bytes:
                raise RuntimeError(f"Incomplete image shard: {image_partial}")
            if intrinsic_partial.stat().st_size != expected_intrinsic_bytes:
                raise RuntimeError(f"Incomplete intrinsic shard: {intrinsic_partial}")
            os.replace(intrinsic_partial, intrinsic_path)
            os.replace(image_partial, image_path)
            checksums[key] = {
                "images_sha256": image_digest.hexdigest(),
                "intrinsics_sha256": intrinsic_digest.hexdigest(),
                "count": shard_size,
            }
            atomic_json(checksums_path, checksums)
            newly_built += shard_size
            elapsed = time.monotonic() - started
            rate = newly_built / elapsed if elapsed > 0 else 0.0
            remaining = count - last
            eta_hours = remaining / rate / 3600 if rate > 0 else float("inf")
            print(
                f"train shard {shard_index + 1}/{shard_count}: "
                f"{last}/{count} images, new_rate={rate:.2f} image/s, "
                f"eta={eta_hours:.2f}h",
                flush=True,
            )
    return checksums


def test_plan(test_root: Path, test_index_path: Path) -> Tuple[List[str], np.ndarray]:
    with test_index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    scenes = sorted(
        set(scene for scene, value in index.items() if value is not None)
        & set(os.listdir(test_root))
    )
    frame_ids = np.asarray(
        [
            index[scene]["context"][:2] + index[scene]["target"][:3]
            for scene in scenes
        ],
        dtype=np.int32,
    )
    return scenes, frame_ids


def build_test_cache(
    *,
    output_dir: Path,
    test_root: Path,
    test_index_path: Path,
    workers: int,
    limit_scenes: int | None,
) -> Tuple[int, str, str, str]:
    scenes, frame_ids = test_plan(test_root, test_index_path)
    if limit_scenes is not None:
        scenes = scenes[:limit_scenes]
        frame_ids = frame_ids[:limit_scenes]
    scene_count = len(scenes)
    atomic_npy(output_dir / "test_scene_names.npy", np.asarray(scenes, dtype=np.bytes_))
    atomic_npy(output_dir / "test_frame_ids.i32.npy", frame_ids)

    image_path = output_dir / "test_images.u8"
    intrinsic_path = output_dir / "test_intrinsics.f32"
    camtoworld_path = output_dir / "test_camtoworlds.f32"
    relative_path_file = output_dir / "test_relative_paths.npy"
    expected_image_bytes = scene_count * 5 * IMAGE_BYTES
    expected_intrinsic_bytes = scene_count * 5 * 3 * 3 * 4
    expected_camtoworld_bytes = scene_count * 5 * 4 * 4 * 4
    if (
        image_path.is_file()
        and intrinsic_path.is_file()
        and camtoworld_path.is_file()
        and relative_path_file.is_file()
        and image_path.stat().st_size == expected_image_bytes
        and intrinsic_path.stat().st_size == expected_intrinsic_bytes
        and camtoworld_path.stat().st_size == expected_camtoworld_bytes
    ):
        return (
            scene_count,
            sha256_file(image_path),
            sha256_file(intrinsic_path),
            sha256_file(camtoworld_path),
        )

    partials = [
        path.with_suffix(path.suffix + ".partial")
        for path in (image_path, intrinsic_path, camtoworld_path)
    ]
    for partial in partials:
        partial.unlink(missing_ok=True)
    relative_paths: List[List[str]] = []
    image_digest = hashlib.sha256()
    intrinsic_digest = hashlib.sha256()
    camtoworld_digest = hashlib.sha256()
    items = [
        (scene, frame_ids[row].tolist(), os.fspath(test_root))
        for row, scene in enumerate(scenes)
    ]
    context = mp.get_context("fork")
    with context.Pool(processes=workers, maxtasksperchild=1024) as pool:
        with partials[0].open("wb", buffering=16 * 1024 * 1024) as image_handle:
            with partials[1].open("wb", buffering=1024 * 1024) as intrinsic_handle:
                with partials[2].open(
                    "wb", buffering=1024 * 1024
                ) as camtoworld_handle:
                    for row, result in enumerate(
                        pool.imap(_build_test_scene, items, chunksize=1), start=1
                    ):
                        images, intrinsics, camtoworlds, paths = result
                        image_bytes = images.tobytes(order="C")
                        intrinsic_bytes = intrinsics.tobytes(order="C")
                        camtoworld_bytes = camtoworlds.tobytes(order="C")
                        image_handle.write(image_bytes)
                        intrinsic_handle.write(intrinsic_bytes)
                        camtoworld_handle.write(camtoworld_bytes)
                        image_digest.update(image_bytes)
                        intrinsic_digest.update(intrinsic_bytes)
                        camtoworld_digest.update(camtoworld_bytes)
                        relative_paths.append(paths)
                        if row % 100 == 0 or row == scene_count:
                            print(f"test scenes: {row}/{scene_count}", flush=True)
                    for handle in (image_handle, intrinsic_handle, camtoworld_handle):
                        handle.flush()
                        os.fsync(handle.fileno())
    expected_sizes = (
        expected_image_bytes,
        expected_intrinsic_bytes,
        expected_camtoworld_bytes,
    )
    for partial, expected_size in zip(partials, expected_sizes):
        if partial.stat().st_size != expected_size:
            raise RuntimeError(f"Incomplete test cache: {partial}")
    atomic_npy(
        relative_path_file,
        np.asarray(relative_paths, dtype=np.bytes_),
    )
    for partial, final in zip(
        partials, (image_path, intrinsic_path, camtoworld_path)
    ):
        os.replace(partial, final)
    return (
        scene_count,
        image_digest.hexdigest(),
        intrinsic_digest.hexdigest(),
        camtoworld_digest.hexdigest(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast-dir",
        required=True,
    )
    parser.add_argument(
        "--schedule",
        required=True,
    )
    parser.add_argument(
        "--data-root",
        required=True,
    )
    parser.add_argument(
        "--test-index",
        default="assets/evaluation_index_re10k.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--train-images-per-shard", type=int, default=4096)
    parser.add_argument("--limit-train-images", type=int)
    parser.add_argument("--limit-test-scenes", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fast_dir = Path(args.fast_dir).resolve()
    schedule_path = Path(args.schedule).resolve()
    data_root = Path(args.data_root).resolve()
    test_index_path = Path(args.test_index).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.train_images_per_shard <= 0:
        raise ValueError("--train-images-per-shard must be positive")
    schedule_sha = sha256_file(schedule_path)
    test_index_sha = sha256_file(test_index_path)
    if schedule_sha != EXPECTED_SCHEDULE_SHA256:
        raise ValueError(f"Unexpected schedule hash: {schedule_sha}")
    if test_index_sha != EXPECTED_TEST_INDEX_SHA256:
        raise ValueError(f"Unexpected test-index hash: {test_index_sha}")
    with (fast_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        source_metadata = json.load(handle)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".build.lock").open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another cache builder owns {output_dir}") from exc

        mapping, globals_in_cache = build_train_plan(
            output_dir, fast_dir, schedule_path
        )
        del mapping
        train_count = (
            len(globals_in_cache)
            if args.limit_train_images is None
            else min(args.limit_train_images, len(globals_in_cache))
        )
        scenes, _ = test_plan(data_root / "test", test_index_path)
        test_scene_count = (
            len(scenes)
            if args.limit_test_scenes is None
            else min(args.limit_test_scenes, len(scenes))
        )
        full_build = (
            args.limit_train_images is None and args.limit_test_scenes is None
        )
        metadata: Dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "build_complete": False,
            "source_dataset_sha256": source_metadata["dataset_sha256"],
            "source_metadata_dir": os.fspath(fast_dir),
            "source_data_root": os.fspath(data_root),
            "train_schedule_sha256": schedule_sha,
            "test_index_sha256": test_index_sha,
            "image_height": 256,
            "image_width": 256,
            "image_channels": 3,
            "image_dtype": "uint8",
            "image_storage": "lossless_raw_uint8_first_use_schedule_shards",
            "resize_contract": "legacy_imageio_rgb_then_opencv_warpAffine",
            "train_input_views": 2,
            "train_supervise_views": 1,
            "test_input_views": 2,
            "test_supervise_views": 3,
            "train_total_source_frames": int(source_metadata["total_frames"]),
            "train_unique_images_full_plan": int(len(globals_in_cache)),
            "train_unique_images": int(train_count),
            "train_images_per_shard": int(args.train_images_per_shard),
            "test_scene_count": int(test_scene_count),
            "test_views_per_scene": 5,
            "limited_audit_build": not full_build,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "imageio_version": getattr(imageio, "__version__", "unknown"),
        }
        atomic_json(output_dir / "metadata.json", metadata)
        free_bytes = shutil.disk_usage(output_dir).free
        required_bytes = train_count * (IMAGE_BYTES + 36) + test_scene_count * (
            5 * (IMAGE_BYTES + 36 + 64)
        )
        if free_bytes < required_bytes + 50_000_000_000:
            raise RuntimeError(
                f"Insufficient free space: free={free_bytes}, required={required_bytes}"
            )
        print(
            f"plan: train={train_count}/{len(globals_in_cache)}, "
            f"test_scenes={test_scene_count}, raw_required={required_bytes / 1e9:.2f}GB, "
            f"free={free_bytes / 1e12:.2f}TB",
            flush=True,
        )
        train_checksums = build_train_shards(
            output_dir=output_dir,
            data_root=data_root / "train",
            fast_dir=fast_dir,
            globals_in_cache=globals_in_cache,
            images_per_shard=args.train_images_per_shard,
            workers=args.workers,
            limit=args.limit_train_images,
        )
        (
            test_scene_count_built,
            test_images_sha,
            test_intrinsics_sha,
            test_camtoworlds_sha,
        ) = build_test_cache(
            output_dir=output_dir,
            test_root=data_root / "test",
            test_index_path=test_index_path,
            workers=args.workers,
            limit_scenes=args.limit_test_scenes,
        )
        if test_scene_count_built != test_scene_count:
            raise RuntimeError("Test cache scene count changed during build.")
        metadata.update(
            {
                "build_complete": bool(full_build),
                "train_shard_count": len(train_checksums),
                "test_images_sha256": test_images_sha,
                "test_intrinsics_sha256": test_intrinsics_sha,
                "test_camtoworlds_sha256": test_camtoworlds_sha,
                "completed_unix_time": time.time(),
            }
        )
        atomic_json(output_dir / "metadata.json", metadata)
        print(
            "FULL CACHE COMPLETE" if full_build else "LIMITED AUDIT CACHE COMPLETE",
            flush=True,
        )


if __name__ == "__main__":
    main()
