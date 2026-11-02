# Data and camera contract

The release keeps the original dataset/model math. Paths are supplied through the
launcher instead of private server defaults. RE10K and Objaverse share the same model implementation with dataset-specific settings.

## RE10K scene/view loader

```text
dataset-root/
  train/<scene>/transforms.json
  train/<scene>/images/<frame>.png
  test/<scene>/transforms.json
  test/<scene>/images/<frame>.png
```

Each transforms JSON contains `w`, `h`, pixel-valued `fl_x`, `fl_y`, `cx`, `cy`,
and `frames`. Each frame contains a relative `file_path` and a 4x4 camera-to-world
`transform_matrix`, as produced by `scripts/gen_transforms.py`. The inherited
loader applies its OpenCV/Blender camera-axis conversion and reference-pose
normalization. The model consumes Blender/OpenGL world-to-camera matrices and
pixel-valued intrinsics. Do not apply the conversion twice.

The metadata utility expects `<data-root>/train`, `<data-root>/test`, and the
original camera text files in `<data-root>/RealEstate10K/train/*.txt` and
`<data-root>/RealEstate10K/test/*.txt`. It reorganizes extracted PNGs into each
scene's `images` subfolder. `data_preprocess.py` creates a separate scene/view
tree with updated metadata and links to the images.

After extracting the source frames and camera metadata, build the loader tree:

```bash
python scripts/gen_transforms.py --data-root /path/to/extracted-re10k
python scripts/data_preprocess.py --data-root /path/to/extracted-re10k \
  --output-root /path/to/re10k --split all
```

The supplied evaluation index is `assets/evaluation_index_re10k.json` (two
context and three target views). Preserve scene/frame ordering when converting
data, because evaluation indices address frames in that ordering.

## RE10K exact legacy sampling

The original main experiment used compact mmap camera metadata plus an immutable
schedule of shape `[2,80001,8,4]`: rank, step, rank-local scene slot, and
`(scene_index, reference_0, reference_1, target)`.

- Schedule SHA-256: `b156b22ff9a8b71c8055faff05d77f2867d31054c6b650a334d6c451a5a2a0d0`
- Default evaluation index SHA-256: `fb695e3aa97bacc61e9d6530cadbe200f410ef93a82773b11b6bd76c736756c5`

The raw-256 cache stores losslessly preprocessed uint8 RGB images for this
sampling stream. The cache builder consumes the compact metadata and existing
schedule; it does not generate the missing compact metadata or recover the
original sampling stream from scratch. The schedule and cache are not included
in the repository. Plain scene/view training is supported, but does not establish
exact reproduction of the reported sampling stream.

## Objaverse paper backend

Use the packed-data loader selected by `--dataset objaverse`. Its metadata contract includes:

- `format_version=1`, `radial_idx=0`, `views_per_object=8`;
- 256x256 images, object-disjoint train/test split;
- `pose_scale=object_scale_ref0_identity_no_unit_baseline`;
- JPEG bytes retained in per-source-shard blobs;
- arrays for object IDs, intrinsics, OpenCV poses, offsets and split indices.

Pack compatible existing rendered shards with:

```bash
python scripts/build_objaverse_radial0_fastdata.py \
  --source-root /path/to/rendered-shards --output-dir /path/to/objaverse-packed
```

The builder packs compatible existing rendered shards. It does not render a new
dataset, download Objaverse assets, or turn an arbitrary Zero123 archive into the
paper dataset. The main training stream uses a fixed two-rank, batch-8 schedule.
Supply `--fast-dir` and its matching `--schedule` for that stream.

Default evaluation uses the packed backend's gap-4 protocol. The paper mixed-gap
protocol uses a different external evaluation wrapper, which is not included in
the preserved source snapshots. The legacy generic Objaverse loader and indices
are retained for their native formats; the release launcher selects the packed
backend explicitly.
