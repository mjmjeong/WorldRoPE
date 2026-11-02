#!/usr/bin/env python3
"""Build a resumable compact Objaverse radial-0 bundle.

The script is read-only with respect to the source dataset.  Existing JPEG
bytes are concatenated without re-encoding, one blob per source shard.  Each
completed shard has an atomic NPZ metadata file and can be reused after an
interruption.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import concurrent.futures
import hashlib
import json
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


FORMAT_VERSION = 1
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_save(path: Path, value: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jpeg_shape(data: bytes) -> tuple[int, int, int]:
    """Read the JPEG SOF marker without doing an expensive pixel decode."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError("missing JPEG SOI marker")
    offset = 2
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[offset : offset + 2])[0]
        if segment_length < 2 or offset + segment_length > len(data):
            raise ValueError("invalid JPEG segment length")
        if marker in sof_markers:
            if segment_length < 8:
                raise ValueError("invalid JPEG SOF segment")
            height = struct.unpack(">H", data[offset + 3 : offset + 5])[0]
            width = struct.unpack(">H", data[offset + 5 : offset + 7])[0]
            components = int(data[offset + 7])
            return height, width, components
        offset += segment_length
    raise ValueError("JPEG SOF marker not found")


def inspect_object(object_dir: Path) -> Dict[str, Any]:
    uid = object_dir.name
    camera_path = object_dir / "cameras.json"
    try:
        if not camera_path.is_file():
            raise FileNotFoundError("missing cameras.json")
        with camera_path.open("r", encoding="utf-8") as handle:
            camera_data = json.load(handle)
        if camera_data.get("object_uid") != uid:
            raise ValueError(
                f"object_uid mismatch: {camera_data.get('object_uid')!r} != {uid!r}"
            )
        cameras = camera_data.get("cameras", [])
        if len(cameras) != 24:
            raise ValueError(f"expected 24 cameras, found {len(cameras)}")
        radial0 = {
            int(camera["view_idx"]): camera
            for camera in cameras
            if int(camera["radial_idx"]) == 0
        }
        if sorted(radial0) != list(range(8)):
            raise ValueError(f"invalid radial-0 view indices: {sorted(radial0)}")

        intrinsics = np.empty((8, 4), dtype=np.float32)
        camtoworlds = np.empty((8, 4, 4), dtype=np.float32)
        image_bytes = []
        for view_index in range(8):
            camera = radial0[view_index]
            expected_name = f"{view_index:03d}_0.jpg"
            if camera["image_name"] != expected_name:
                raise ValueError(
                    f"view {view_index} image_name {camera['image_name']!r} "
                    f"!= {expected_name!r}"
                )
            intrinsic = camera["intrinsics"]
            if list(map(int, intrinsic["image_size"])) != [256, 256]:
                raise ValueError(
                    f"view {view_index} image_size={intrinsic['image_size']}"
                )
            fx, fy = map(float, intrinsic["focal_length"])
            cx, cy = map(float, intrinsic["principal_point"])
            if min(fx, fy) <= 0:
                raise ValueError(f"view {view_index} non-positive focal length")
            intrinsics[view_index] = [fx, fy, cx, cy]

            c2w_3x4 = np.asarray(
                camera["extrinsics"]["camera_to_world"], dtype=np.float64
            )
            w2c_3x4 = np.asarray(
                camera["extrinsics"]["world_to_camera"], dtype=np.float64
            )
            if c2w_3x4.shape != (3, 4) or w2c_3x4.shape != (3, 4):
                raise ValueError(f"view {view_index} invalid extrinsic shape")
            c2w_blender = np.eye(4, dtype=np.float64)
            w2c_blender = np.eye(4, dtype=np.float64)
            c2w_blender[:3] = c2w_3x4
            w2c_blender[:3] = w2c_3x4
            inverse_error = float(
                np.max(np.abs(w2c_blender @ c2w_blender - np.eye(4)))
            )
            if inverse_error > 2e-5:
                raise ValueError(
                    f"view {view_index} inverse extrinsic error={inverse_error}"
                )
            c2w_opencv = c2w_blender @ BLENDER_TO_OPENCV
            world_origin_camera = np.linalg.inv(c2w_opencv) @ np.array(
                [0.0, 0.0, 0.0, 1.0]
            )
            if world_origin_camera[2] <= 0:
                raise ValueError(
                    f"view {view_index} object origin is behind OpenCV camera"
                )
            camtoworlds[view_index] = c2w_opencv.astype(np.float32)

            image_path = object_dir / "views" / expected_name
            data = image_path.read_bytes()
            height, width, components = jpeg_shape(data)
            if (height, width, components) != (256, 256, 3):
                raise ValueError(
                    f"view {view_index} JPEG shape="
                    f"{(height, width, components)}"
                )
            image_bytes.append(data)
        return {
            "valid": True,
            "uid": uid,
            "intrinsics": intrinsics,
            "camtoworlds": camtoworlds,
            "image_bytes": image_bytes,
        }
    except Exception as exc:
        return {
            "valid": False,
            "uid": uid,
            "path": str(object_dir),
            "error": repr(exc),
        }


def write_shard(
    shard_dir: Path, staging: Path, workers: int
) -> Dict[str, Any]:
    shard_name = shard_dir.name
    npz_path = staging / "shards" / f"{shard_name}.npz"
    blob_path = staging / "blobs" / f"{shard_name}.jpgblob"
    done_path = staging / "shards" / f"{shard_name}.done.json"
    if npz_path.is_file() and blob_path.is_file() and done_path.is_file():
        with done_path.open("r", encoding="utf-8") as handle:
            done = json.load(handle)
        if (
            int(done.get("format_version", -1)) == FORMAT_VERSION
            and int(done.get("blob_bytes", -1)) == blob_path.stat().st_size
        ):
            return done

    object_dirs = sorted(path for path in shard_dir.iterdir() if path.is_dir())
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(inspect_object, object_dirs))

    valid = [result for result in results if result["valid"]]
    invalid = [result for result in results if not result["valid"]]
    uids = np.asarray([result["uid"].encode("utf-8") for result in valid], dtype="S64")
    intrinsics = np.stack([result["intrinsics"] for result in valid])
    camtoworlds = np.stack([result["camtoworlds"] for result in valid])
    offsets = np.empty((len(valid), 8, 2), dtype=np.uint64)

    blob_tmp = blob_path.with_suffix(blob_path.suffix + ".tmp")
    digest = hashlib.sha256()
    cursor = 0
    with blob_tmp.open("wb") as blob:
        for object_index, result in enumerate(valid):
            for view_index, data in enumerate(result["image_bytes"]):
                offsets[object_index, view_index] = [cursor, len(data)]
                blob.write(data)
                digest.update(data)
                cursor += len(data)
        blob.flush()
        os.fsync(blob.fileno())
    os.replace(blob_tmp, blob_path)

    npz_tmp = npz_path.with_suffix(npz_path.suffix + ".tmp")
    with npz_tmp.open("wb") as handle:
        np.savez(
            handle,
            uids=uids,
            intrinsics=intrinsics,
            camtoworlds=camtoworlds,
            jpeg_offsets=offsets,
        )
    os.replace(npz_tmp, npz_path)
    done = {
        "format_version": FORMAT_VERSION,
        "shard": shard_name,
        "object_directories": len(object_dirs),
        "valid_objects": len(valid),
        "invalid_objects": len(invalid),
        "invalid_records": invalid,
        "blob_bytes": cursor,
        "blob_sha256": digest.hexdigest(),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(done_path, done)
    return done


def concatenate_metadata(staging: Path, shard_names: list[str]) -> Dict[str, np.ndarray]:
    uids = []
    shard_indices = []
    intrinsics = []
    camtoworlds = []
    jpeg_offsets = []
    for shard_index, shard_name in enumerate(shard_names):
        with np.load(
            staging / "shards" / f"{shard_name}.npz", allow_pickle=False
        ) as shard:
            shard_uids = shard["uids"]
            uids.append(shard_uids)
            shard_indices.append(
                np.full(len(shard_uids), shard_index, dtype=np.uint16)
            )
            intrinsics.append(shard["intrinsics"])
            camtoworlds.append(shard["camtoworlds"])
            jpeg_offsets.append(shard["jpeg_offsets"])
    return {
        "object_uids": np.concatenate(uids),
        "shard_indices": np.concatenate(shard_indices),
        "intrinsics": np.concatenate(intrinsics),
        "camtoworlds": np.concatenate(camtoworlds),
        "jpeg_offsets": np.concatenate(jpeg_offsets),
        "shard_names": np.asarray(
            [name.encode("utf-8") for name in shard_names], dtype="S7"
        ),
    }


def domain_rng(seed: int, domain: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, domain])))


def build_splits_and_schedule(
    staging: Path, arrays: Dict[str, np.ndarray], split_seed: int
) -> Dict[str, Any]:
    object_count = len(arrays["object_uids"])
    if object_count < 66_033 + 6_474:
        raise RuntimeError(
            f"Only {object_count} eligible objects; need at least 72507."
        )
    permutation = domain_rng(split_seed, 0x53504C49).permutation(object_count)
    train_indices = np.sort(permutation[:66_033]).astype(np.int32)
    test_indices = np.sort(permutation[66_033 : 66_033 + 6_474]).astype(np.int32)
    reserve_indices = np.sort(permutation[66_033 + 6_474 :]).astype(np.int32)
    if np.intersect1d(train_indices, test_indices).size:
        raise AssertionError("Train/test split overlap")

    triplets = np.asarray(
        [
            (start, (start + gap) % 8, (start + target_offset) % 8)
            for start in range(8)
            for gap in (2, 3, 4)
            for target_offset in range(1, gap)
        ],
        dtype=np.int32,
    )
    if triplets.shape != (48, 3):
        raise AssertionError(triplets.shape)
    schedule_rng = domain_rng(split_seed, 0x53434844)
    object_choices = train_indices[
        schedule_rng.integers(0, len(train_indices), size=(80_001, 16))
    ]
    triplet_choices = triplets[
        schedule_rng.integers(0, len(triplets), size=(80_001, 16))
    ]
    global_schedule = np.concatenate(
        [object_choices[..., None], triplet_choices], axis=-1
    ).astype(np.int32)
    schedule = np.stack(
        [global_schedule[:, :8], global_schedule[:, 8:]], axis=0
    )

    test_views = np.empty((len(test_indices), 5), dtype=np.uint8)
    for position, object_index in enumerate(test_indices):
        uid = bytes(arrays["object_uids"][object_index]).decode("utf-8")
        digest = hashlib.blake2b(
            f"objaverse-test-arc-v1:{split_seed}:{uid}".encode("utf-8"),
            digest_size=8,
        ).digest()
        start = int.from_bytes(digest, byteorder="little") % 8
        test_views[position] = [
            start,
            (start + 4) % 8,
            (start + 1) % 8,
            (start + 2) % 8,
            (start + 3) % 8,
        ]

    atomic_save(staging / "train_object_indices.i32.npy", train_indices)
    atomic_save(staging / "test_object_indices.i32.npy", test_indices)
    atomic_save(staging / "reserve_object_indices.i32.npy", reserve_indices)
    atomic_save(staging / "test_views.u8.npy", test_views)
    schedule_path = staging / "train_schedule_seed42_2rank_b8_80k_v1.npy"
    atomic_save(schedule_path, schedule)
    return {
        "train_count": len(train_indices),
        "test_count": len(test_indices),
        "reserve_count": len(reserve_indices),
        "train_triplets_per_object": len(triplets),
        "train_schedule_shape": list(schedule.shape),
        "train_schedule_sha256": sha256_file(schedule_path),
        "test_view_shape": list(test_views.shape),
        "test_target_images": len(test_indices) * 3,
    }


def dataset_digest(arrays: Dict[str, np.ndarray], shard_done: list[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for key in ("object_uids", "shard_indices", "intrinsics", "camtoworlds", "jpeg_offsets"):
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(arrays[key]).view(np.uint8))
    for done in shard_done:
        digest.update(done["shard"].encode("utf-8"))
        digest.update(done["blob_sha256"].encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    repo_root = Path(__file__).resolve().parent.parent
    builder_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=repo_root,
        text=True,
    ).strip():
        raise RuntimeError("Refusing to build from a dirty tracked worktree.")
    source_root = args.source_root.resolve()
    output_dir = args.output_dir
    staging = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite completed bundle: {output_dir}")
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "blobs").mkdir(exist_ok=True)
    (staging / "shards").mkdir(exist_ok=True)
    shard_dirs = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and path.name.startswith("000-")
    )
    if len(shard_dirs) != 160:
        raise RuntimeError(f"Expected 160 source shards, found {len(shard_dirs)}")

    print(
        f"build_start source={source_root} output={output_dir} "
        f"workers={args.workers} shards={len(shard_dirs)}",
        flush=True,
    )
    shard_done = []
    for shard_number, shard_dir in enumerate(shard_dirs, 1):
        done = write_shard(shard_dir, staging, args.workers)
        shard_done.append(done)
        print(
            f"shard={shard_dir.name} progress={shard_number}/{len(shard_dirs)} "
            f"valid={done['valid_objects']} invalid={done['invalid_objects']} "
            f"blob_mb={done['blob_bytes']/1e6:.2f} "
            f"elapsed_total_min={(time.time()-started)/60:.2f}",
            flush=True,
        )

    shard_names = [path.name for path in shard_dirs]
    arrays = concatenate_metadata(staging, shard_names)
    object_count = len(arrays["object_uids"])
    if len(set(arrays["object_uids"].tolist())) != object_count:
        raise RuntimeError("Duplicate object UID in eligible set")
    atomic_save(staging / "object_uids.npy", arrays["object_uids"])
    atomic_save(staging / "shard_indices.npy", arrays["shard_indices"])
    atomic_save(staging / "intrinsics.f32.npy", arrays["intrinsics"])
    atomic_save(
        staging / "camtoworlds_opencv.f32.npy", arrays["camtoworlds"]
    )
    atomic_save(staging / "jpeg_offsets.u64.npy", arrays["jpeg_offsets"])
    atomic_save(staging / "shard_names.npy", arrays["shard_names"])
    split_info = build_splits_and_schedule(staging, arrays, args.split_seed)

    invalid_records = [
        record
        for done in shard_done
        for record in done.get("invalid_records", [])
    ]
    atomic_json(staging / "invalid_objects.json", invalid_records)
    manifest = {
        "format_version": FORMAT_VERSION,
        "builder_git_commit": builder_commit,
        "source_root": str(source_root),
        "radial_idx": 0,
        "views_per_object": 8,
        "image_width": 256,
        "image_height": 256,
        "image_storage": "raw_jpeg_bytes_in_per_source_shard_blobs",
        "jpeg_reencoded": False,
        "camera_coordinate_convention": "OpenCV_c2w_x_right_y_down_z_forward",
        "pose_scale": "object_scale_ref0_identity_no_unit_baseline",
        "object_sampling": "uniform_object_then_uniform_directed_triplet",
        "train_context_cyclic_gaps": [2, 3, 4],
        "test_context_cyclic_gap": 4,
        "fov_filter": None,
        "camera_distance_filter": None,
        "split_seed": args.split_seed,
        "model_seed_independent_schedule": True,
        "source_object_directory_count": sum(
            int(done["object_directories"]) for done in shard_done
        ),
        "eligible_object_count": object_count,
        "invalid_object_count": len(invalid_records),
        "source_shard_count": len(shard_names),
        "dataset_sha256": dataset_digest(arrays, shard_done),
        "elapsed_seconds": time.time() - started,
        **split_info,
    }
    atomic_json(staging / "metadata.json", manifest)
    os.replace(staging, output_dir)
    print(
        f"build_complete output={output_dir} eligible={object_count} "
        f"invalid={len(invalid_records)} reserve={split_info['reserve_count']} "
        f"elapsed_min={(time.time()-started)/60:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
