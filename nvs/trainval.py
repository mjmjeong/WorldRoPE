# Copyright (c) 2026 Applied Intuition, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_repo_root_s = str(_repo_root)
if _repo_root_s not in sys.path:
    sys.path.insert(0, _repo_root_s)

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from einops import rearrange
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nvs.dataset import EvalDataset, TrainDataset
from nvs.dataset_objaverse import ObjaverseTrainDataset, ObjaverseEvalDataset
from nvs.objaverse_fast_dataset import (
    ObjaverseFastEvalDataset,
    ObjaverseFixedScheduleDataset,
)
from nvs.re10k_fast_dataset import LegacyScheduleDataset, URoPEFastTrainDataset
from nvs.re10k_raw_cache import RE10KRawCachedEvalDataset
from nvs.lvsm import (
    Camera,
    LVSMDecoderOnlyModel,
    LVSMDecoderOnlyModelConfig,
)
from nvs.perceptual import Perceptual
from pos_enc.utils.functional import random_SO3
from pos_enc.utils.data_rng import (
    DeterministicSampleDataset,
    FixedScheduleBatchSampler,
    LegacyScheduleBatchSampler,
    ResumableSequentialSampler,
    make_dataloader_generator,
    seed_dataloader_worker,
)
from pos_enc.utils.runner import Launcher, LauncherConfig, nested_to_device


def write_tensor_to_image(
    x: Tensor,
    path: str,
    downscale: int = 1,
    sqrt: bool = False,
    point: Tuple[int, int] = None,
):
    # x: [H, W, 1 or 3] in (0, 1)
    assert x.ndim == 3, x.shape
    if x.shape[-1] == 1:
        x = x.repeat(1, 1, 3)
    if sqrt:
        # reshape image to square
        h, w = x.shape[:2]
        if h > w:
            n_images = h // w
            n_sqrt = int(np.sqrt(n_images))
            x = rearrange(x, "(n1 n2 h) w c -> (n1 h) (n2 w) c", n1=n_sqrt, n2=n_sqrt)
        elif h < w:
            n_images = w // h
            n_sqrt = int(np.sqrt(n_images))
            x = rearrange(x, "h (n1 n2 w) c -> (n1 h) (n2 w) c", n1=n_sqrt, n2=n_sqrt)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = (x * 255).to(torch.uint8).detach().cpu().numpy()
    if downscale > 1:
        image = cv2.resize(image, (0, 0), fx=1.0 / downscale, fy=1.0 / downscale)
    if point is not None:
        cv2.circle(image, point, 5, (255, 0, 0), -1)
    imageio.imsave(path, image)


@dataclass
class LVSMLauncherConfig(LauncherConfig):
    # Dataset config
    dataset_name: str = "realestate10k"
    dataset_patch_size: int = 256
    dataset_supervise_views: int = 6
    dataset_batch_scenes: int = 4
    # Optional compact mmap camera metadata; the release requires explicit paths.
    re10k_fast_train_dir: Optional[str] = None
    # Exact scene/context/target selections recovered from the eight legacy
    # seed-42, 2-rank, batch-8 runs.  It is one mmap array and therefore adds
    # no per-sample JSON or schedule parsing.
    re10k_legacy_schedule_file: Optional[str] = None
    # Objaverse radial-0 uses an object-disjoint seed-42 split, compact mmap
    # cameras, packed source JPEG bytes, and one immutable 2-rank schedule.
    objaverse_fast_dir: Optional[str] = None
    objaverse_schedule_file: Optional[str] = None
    # Lossless uint8 payloads produced by the exact legacy image preprocessing
    # path. Sampling and pose normalization remain unchanged.
    re10k_image_cache_dir: Optional[str] = None
    # This stream is deliberately separate from the model seed so changing PE
    # type, heads or learned anchor modules cannot change training samples.
    data_seed: int = 42
    # Four workers per DDP rank keeps a 2-GPU/10-CPU Run below CPU
    # oversubscription while prefetching one full per-rank batch.
    train_num_workers: int = 4
    train_prefetch_factor: int = 2
    train_zoom_factor: float = 1.0
    random_zoom: bool = False

    # Optimization config
    use_torch_compile: bool = True

    # Model config
    model_config: Any = field(
        default_factory=lambda: LVSMDecoderOnlyModelConfig(ref_views=2)
    )

    # Training config
    max_steps: int = 100_000  # override
    ckpt_every: int = 1000  # override
    print_every: int = 100
    visual_every: int = 100
    lr: float = 4e-4
    warmup_steps: int = 2500

    # perceptual loss weight.
    perceptual_loss_w: float = 0.5

    # How many test scenes to run.
    # Full RE10K evaluation is expensive.  Future 80k training runs evaluate
    # once, at the final step, unless a diagnostic config explicitly overrides it.
    test_every: int = 80000
    test_n: Optional[int] = None
    test_input_views: int = 2
    test_supervise_views: int = 3
    test_zoom_factor: tuple[float, ...] = (1.0,)
    aug_with_world_origin_shift: bool = False
    aug_with_world_rotation: bool = False

    # Render a video
    render_video: bool = False
    render_views: bool = False

    # test index file
    test_index_fp: Optional[str] = None


class LVSMLauncher(Launcher):
    config: LVSMLauncherConfig

    # Data preprocessing.
    def preprocess(
        self, data: Dict, input_views: int
    ) -> Tuple[Tensor, Camera, Camera, Tensor]:
        data = nested_to_device(data, self.device)

        images = data["image"] / 255.0
        Ks = data["K"]
        camtoworlds = data["camtoworld"]
        image_paths = data["image_path"]
        assert images.ndim == 5, images.shape
        n_batch, n_views, height, width, _ = images.shape

        # random shift and rotate.
        aug = torch.eye(4, device=self.device).repeat(n_batch, 1, 1)
        if self.config.aug_with_world_origin_shift:
            shifts = torch.randn((n_batch, 3), device=self.device)
            aug[:, :3, 3] = shifts
        if self.config.aug_with_world_rotation:
            rotations = random_SO3((n_batch,), device=self.device)
            aug[:, :3, :3] = rotations
        camtoworlds = torch.einsum("bij,bvjk->bvik", aug, camtoworlds)

        ref_imgs = images[:, :input_views]
        tar_imgs = images[:, input_views:]
        ref_cams = Camera(
            K=Ks[:, :input_views],
            camtoworld=camtoworlds[:, :input_views],
            width=width,
            height=height,
        )
        tar_cams = Camera(
            K=Ks[:, input_views:],
            camtoworld=camtoworlds[:, input_views:],
            width=width,
            height=height,
        )
        ref_paths = np.array(image_paths)[:input_views]
        tar_paths = np.array(image_paths)[input_views:]

        processed = {
            "ref_imgs": ref_imgs,
            "tar_imgs": tar_imgs,
            "ref_cams": ref_cams,
            "tar_cams": tar_cams,
            "ref_paths": ref_paths,
            "tar_paths": tar_paths,
        }
        return processed

    def train_initialize(self) -> Dict[str, Any]:
        # ------------- Setup Data. ------------- #
        train_data_sampler = None
        train_batch_sampler = None
        if self.config.dataset_name == "realestate10k":
            train_root = "./data_processed/realestate10k/train"
            if self.config.re10k_fast_train_dir is not None:
                dataset = URoPEFastTrainDataset(
                    data_root=train_root,
                    fast_dir=self.config.re10k_fast_train_dir,
                    patch_size=self.config.dataset_patch_size,
                    zoom_factor=self.config.train_zoom_factor,
                    random_zoom=self.config.random_zoom,
                    supervise_views=self.config.dataset_supervise_views,
                    image_cache_dir=self.config.re10k_image_cache_dir,
                )
            else:
                scenes = sorted(glob.glob(f"{train_root}/*"))
                dataset = TrainDataset(
                    scenes,
                    patch_size=self.config.dataset_patch_size,
                    zoom_factor=self.config.train_zoom_factor,
                    random_zoom=self.config.random_zoom,
                    supervise_views=self.config.dataset_supervise_views,
                )
            source_scene_count = len(dataset)
            if self.config.re10k_legacy_schedule_file is not None:
                if not isinstance(dataset, URoPEFastTrainDataset):
                    raise ValueError(
                        "The recovered legacy schedule requires the compact "
                        "RE10K metadata backend."
                    )
                if self.world_size != 2:
                    raise ValueError(
                        "The recovered legacy schedule requires world_size=2."
                    )
                if self.config.dataset_batch_scenes != 8 or self.config.acc != 1:
                    raise ValueError(
                        "The recovered legacy schedule requires batch=8 per "
                        "rank and acc=1."
                    )
                dataset = LegacyScheduleDataset(
                    dataset=dataset,
                    schedule_path=self.config.re10k_legacy_schedule_file,
                    world_rank=self.world_rank,
                    max_steps=self.config.max_steps,
                )
                train_batch_sampler = LegacyScheduleBatchSampler(
                    max_steps=self.config.max_steps,
                    scene_count=source_scene_count,
                    batch_size=self.config.dataset_batch_scenes,
                )
            else:
                dataset = DeterministicSampleDataset(
                    dataset, self.config.data_seed, self.world_rank
                )
                train_data_sampler = ResumableSequentialSampler(len(dataset))
        else:
            assert self.config.dataset_name == "objaverse"
            if self.config.objaverse_fast_dir is not None:
                if self.config.objaverse_schedule_file is None:
                    raise ValueError(
                        "Objaverse fastdata requires objaverse_schedule_file."
                    )
                if self.world_size != 2:
                    raise ValueError(
                        "The fixed Objaverse schedule requires world_size=2."
                    )
                if (
                    self.config.dataset_batch_scenes != 8
                    or self.config.acc != 1
                    or self.config.dataset_supervise_views != 1
                ):
                    raise ValueError(
                        "The fixed Objaverse schedule requires batch=8 per rank, "
                        "acc=1 and one train target."
                    )
                dataset = ObjaverseFixedScheduleDataset(
                    fast_dir=self.config.objaverse_fast_dir,
                    schedule_path=self.config.objaverse_schedule_file,
                    world_rank=self.world_rank,
                    max_steps=self.config.max_steps,
                    patch_size=self.config.dataset_patch_size,
                    zoom_factor=self.config.train_zoom_factor,
                    random_zoom=self.config.random_zoom,
                )
                source_scene_count = int(dataset.metadata["train_count"])
                train_batch_sampler = FixedScheduleBatchSampler(
                    max_steps=self.config.max_steps,
                    batch_size=self.config.dataset_batch_scenes,
                )
            else:
                train_index_file = "assets/objaverse_train.json"
                dataset = ObjaverseTrainDataset(
                    train_index_file=train_index_file,
                    patch_size=self.config.dataset_patch_size,
                    zoom_factor=self.config.train_zoom_factor,
                    random_zoom=self.config.random_zoom,
                    supervise_views=self.config.dataset_supervise_views,
                )
                source_scene_count = len(dataset)
        num_workers = int(self.config.train_num_workers)
        if num_workers < 0:
            raise ValueError("train_num_workers must be non-negative.")
        if self.config.train_prefetch_factor <= 0:
            raise ValueError("train_prefetch_factor must be positive.")
        dataloader_kwargs = dict(
            dataset=dataset,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        if train_batch_sampler is not None:
            dataloader_kwargs["batch_sampler"] = train_batch_sampler
            dataloader_kwargs["generator"] = make_dataloader_generator(
                self.config.data_seed, self.world_rank
            )
        else:
            dataloader_kwargs["batch_size"] = self.config.dataset_batch_scenes
        if train_data_sampler is not None:
            dataloader_kwargs.update(
                sampler=train_data_sampler,
                worker_init_fn=seed_dataloader_worker,
                generator=make_dataloader_generator(
                    self.config.data_seed, self.world_rank
                ),
            )
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = int(
                self.config.train_prefetch_factor
            )
        dataloader = torch.utils.data.DataLoader(**dataloader_kwargs)
        self.logging_on_master(f"Total scenes: {source_scene_count}")
        if self.config.dataset_name == "realestate10k":
            base_dataset = dataset.dataset
            if isinstance(dataset, LegacyScheduleDataset):
                self.logging_on_master(
                    "Legacy-exact compact URoPE RE10K stream enabled: "
                    f"fast_dir={base_dataset.fast_dir}, "
                    f"schedule={dataset.schedule_path}, "
                    f"image_cache={base_dataset.image_cache_dir}, "
                    f"dataset_sha256={base_dataset.metadata['dataset_sha256']}, "
                    f"rank={self.world_rank}, legacy_seed=42, "
                    "legacy_workers=16, legacy_batch=8."
                )
            if isinstance(base_dataset, URoPEFastTrainDataset):
                if not isinstance(dataset, LegacyScheduleDataset):
                    self.logging_on_master(
                        "Compact URoPE RE10K metadata enabled: "
                        f"fast_dir={base_dataset.fast_dir}, "
                        f"dataset_sha256={base_dataset.metadata['dataset_sha256']}, "
                        f"data_seed={self.config.data_seed}, rank={self.world_rank}, "
                        "stateless_sample_index=true."
                    )
            else:
                self.logging_on_master(
                    "Deterministic URoPE RE10K JSON audit stream enabled: "
                    f"data_seed={self.config.data_seed}, rank={self.world_rank}, "
                    "stateless_sample_index=true."
                )
        elif isinstance(dataset, ObjaverseFixedScheduleDataset):
            self.logging_on_master(
                "Fixed compact Objaverse radial-0 stream enabled: "
                f"fast_dir={dataset.fast_dir}, "
                f"schedule={dataset.schedule_path}, "
                f"dataset_sha256={dataset.metadata['dataset_sha256']}, "
                f"schedule_sha256={dataset.metadata['train_schedule_sha256']}, "
                f"rank={self.world_rank}, split_seed=42, "
                "directed_gaps=2,3,4, packed_jpeg=true."
            )

        # ------------- Setup Model. ------------- #
        model = LVSMDecoderOnlyModel(self.config.model_config).to(self.device)
        # Apply torch.compile for performance optimization if enabled
        if self.config.use_torch_compile:
            model = torch.compile(model)
        perceptual = Perceptual().to(self.device)
        print(f"Model is initialized in rank {self.world_rank}")

        # ------------- Setup Optimizer. ------------- #
        # Paper A.1 "We use a weight decay of 0.05 on all parameters except
        # the weights of LayerNorm layers."
        def no_weight_decay(name: str) -> bool:
            return (
                "norm" in name
                or "mix_weight" in name
                or "mixed_freqs" in name
            )

        params_decay = {
            "params": [
                p for n, p in model.named_parameters() if not no_weight_decay(n)
            ],
            "weight_decay": 0.5,
        }
        params_no_decay = {
            "params": [
                p for n, p in model.named_parameters() if no_weight_decay(n)
            ],
            "weight_decay": 0.0,
        }
        optimizer = torch.optim.AdamW(
            [params_decay, params_no_decay], lr=self.config.lr, betas=(0.9, 0.95)
        )

        # ------------- Setup Scheduler. ------------- #
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,
                    total_iters=self.config.warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.max_steps - self.config.warmup_steps,
                ),
            ]
        )

        # ------------- Setup Metrics. ------------- #
        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        # Note: careful when comparing with papers: "vgg" or "alex"
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

        # prepare returns
        state = {
            "model": model,
            "perceptual": perceptual,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "dataloader": dataloader,
            # The runner creates this iterator after checkpoint loading and
            # positions the logical sampler at the exact resume offset.
            "dataiter": (
                None
                if train_data_sampler is not None or train_batch_sampler is not None
                else iter(dataloader)
            ),
            "train_data_sampler": train_data_sampler,
            "train_batch_sampler": train_batch_sampler,
            "ssim_fn": ssim_fn,
            "psnr_fn": psnr_fn,
            "lpips_fn": lpips_fn,
        }
        print(f"Launcher(train) is intialized in rank {self.world_rank}")
        return state

    def train_iteration(
        self, step: int, state: Dict[str, Any], acc_step: int, *args, **kwargs
    ) -> None:
        dataloader = state["dataloader"]
        dataiter = state["dataiter"]
        perceptual = state["perceptual"]
        model = state["model"]
        model.train()

        try:
            data = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            data = next(dataiter)
            state["dataiter"] = dataiter

        input_views = data["K"].shape[1] - self.config.dataset_supervise_views
        processed = self.preprocess(data, input_views=input_views)
        ref_imgs, tar_imgs = processed["ref_imgs"], processed["tar_imgs"]
        ref_cams, tar_cams = processed["ref_cams"], processed["tar_cams"]

        # Forward.
        with torch.amp.autocast("cuda", enabled=self.config.amp, dtype=self.amp_dtype):
            outputs = model(ref_imgs, ref_cams, tar_cams)
            outputs = torch.sigmoid(outputs)
            mse = F.mse_loss(outputs, tar_imgs)

            if self.config.perceptual_loss_w > 0:
                perceptual_loss = perceptual(
                    rearrange(outputs, "b v h w c -> (b v) c h w"),
                    rearrange(tar_imgs, "b v h w c -> (b v) c h w"),
                )
                loss = mse + perceptual_loss * self.config.perceptual_loss_w
            else:
                loss = mse

        # The opt-in non-finite policy is handled by the distributed runner,
        # but training metrics (especially LPIPS) require finite images.  Bail
        # out before any visualization or metric code so every rank can
        # synchronize and skip the same update cleanly.
        if self.config.skip_nonfinite_updates and not bool(
            torch.isfinite(outputs.detach()).all().item()
            and torch.isfinite(loss.detach()).all().item()
        ):
            raise FloatingPointError("non-finite model output or loss")

        # Loggings.
        if (
            self.config.visual_every > 0
            and step % self.config.visual_every == 0
            and self.world_rank == 0
            and acc_step == 0
        ):
            write_tensor_to_image(
                rearrange(outputs, "b v h w c-> (b h) (v w) c"),
                f"{self.visual_dir}/outputs.png",
            )
            write_tensor_to_image(
                rearrange(tar_imgs, "b v h w c-> (b h) (v w) c"),
                f"{self.visual_dir}/gts.png",
            )
            write_tensor_to_image(
                rearrange(ref_imgs, "b v h w c-> (b h) (v w) c"),
                f"{self.visual_dir}/inputs.png",
            )

        if (
            step % self.config.print_every == 0
            and self.world_rank == 0
            and acc_step == 0
        ):
            mse = F.mse_loss(outputs, tar_imgs)
            outputs = rearrange(outputs, "b v h w c-> (b v) c h w")
            tar_imgs = rearrange(tar_imgs, "b v h w c-> (b v) c h w")
            psnr = state["psnr_fn"](outputs, tar_imgs)
            ssim = state["ssim_fn"](outputs, tar_imgs)
            lpips = state["lpips_fn"](outputs, tar_imgs)
            self.logging_on_master(
                f"Step: {step}, Loss: {loss:.3f}, PSNR: {psnr:.3f}, "
                f"SSIM: {ssim:.3f}, LPIPS: {lpips:.3f}, "
                f"LR: {state['scheduler'].get_last_lr()[0]:.3e}"
            )
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/psnr", psnr, step)
            self.writer.add_scalar("train/ssim", ssim, step)
            self.writer.add_scalar("train/lpips", lpips, step)
        return loss

    def test_initialize(
        self,
        model: Optional[torch.nn.Module] = None,
    ) -> Dict[str, Any]:
        # ------------- Setup Data. ------------- #
        dataset = None
        dataloaders = dict()
        if not self.config.render_video and self.config.test_index_fp is None:
            assert (
                self.config.test_input_views == 2
                and self.config.test_supervise_views == 3
            ), "Invalid input views and supervise views for RE10K, should be 2 and 3 respectively."
        folder = "./data_processed/realestate10k/test/"
        for zoom_factor in self.config.test_zoom_factor:
            if self.config.dataset_name == "realestate10k":
                eval_dataset_cls = (
                    RE10KRawCachedEvalDataset
                    if self.config.re10k_image_cache_dir is not None
                    else EvalDataset
                )
                eval_dataset_kwargs = dict(
                    folder=folder,
                    patch_size=self.config.dataset_patch_size,
                    zoom_factor=zoom_factor,
                    first_n=self.config.test_n,
                    rank=self.world_rank,
                    world_size=self.world_size,
                    input_views=self.config.test_input_views,
                    supervise_views=self.config.test_supervise_views,
                    render_video=self.config.render_video,
                    test_index_fp=self.config.test_index_fp,
                )
                if self.config.re10k_image_cache_dir is not None:
                    eval_dataset_kwargs["image_cache_dir"] = (
                        self.config.re10k_image_cache_dir
                    )
                dataset = eval_dataset_cls(**eval_dataset_kwargs)
            elif self.config.dataset_name == "objaverse":
                if self.config.objaverse_fast_dir is not None:
                    dataset = ObjaverseFastEvalDataset(
                        fast_dir=self.config.objaverse_fast_dir,
                        patch_size=self.config.dataset_patch_size,
                        zoom_factor=zoom_factor,
                        first_n=self.config.test_n,
                        rank=self.world_rank,
                        world_size=self.world_size,
                        input_views=self.config.test_input_views,
                        supervise_views=self.config.test_supervise_views,
                        render_video=self.config.render_video,
                        test_index_fp=self.config.test_index_fp,
                    )
                else:
                    test_index_file = "assets/objaverse_test.json"
                    dataset = ObjaverseEvalDataset(
                        test_index_file=test_index_file,
                        patch_size=self.config.dataset_patch_size,
                        zoom_factor=zoom_factor,
                        first_n=self.config.test_n,
                        rank=self.world_rank,
                        world_size=self.world_size,
                        input_views=self.config.test_input_views,
                        supervise_views=self.config.test_supervise_views,
                        render_video=self.config.render_video,
                        test_index_fp=self.config.test_index_fp,
                    )
            dataloaders[f"zoom{zoom_factor}"] = (
                self.config.test_input_views,
                torch.utils.data.DataLoader(
                    dataset, batch_size=1, num_workers=16, pin_memory=True
                ),
            )
        self.logging_on_master(f"Total scenes: {len(dataset)}")

        # ------------- Setup Model. ------------- #
        if model is None:
            model = LVSMDecoderOnlyModel(self.config.model_config).to(self.device)
            # Apply torch.compile for performance optimization if enabled
            if self.config.use_torch_compile:
                model = torch.compile(model)
            print(f"Model is initialized in rank {self.world_rank}")

        # ------------- Setup Metrics. ------------- #
        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        # Note: careful when comparing with papers: "vgg" or "alex"
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

        # prepare returns
        state = {
            "model": model,
            "dataloaders": dataloaders,
            "psnr_fn": psnr_fn,
            "ssim_fn": ssim_fn,
            "lpips_fn": lpips_fn,
        }
        print(f"Launcher(Test) is intialized in rank {self.world_rank}")
        return state

    @torch.inference_mode()
    def test_iteration(self, step: int, state: Dict[str, Any]) -> None:
        dataloaders = state["dataloaders"]
        model = state["model"]
        model.eval()

        for label, (input_views, dataloader) in dataloaders.items():
            psnrs, lpips, ssims = [], [], []
            canvas = []  # for visualization
            for iter_idx, data in tqdm.tqdm(enumerate(dataloader), desc="Testing"):
                processed = self.preprocess(data, input_views=input_views)
                ref_imgs, tar_imgs = processed["ref_imgs"], processed["tar_imgs"]
                ref_cams, tar_cams = processed["ref_cams"], processed["tar_cams"]
                ref_paths, tar_paths = processed["ref_paths"], processed["tar_paths"]
                # Forward.
                with torch.amp.autocast(
                    "cuda", enabled=self.config.amp, dtype=self.amp_dtype
                ):
                    outputs = model(ref_imgs, ref_cams, tar_cams)
                    outputs = torch.sigmoid(outputs)

                if self.config.render_video:
                    assert outputs.shape[0] == 1
                    path_splits = tar_paths[0, 0].split("/")
                    scene_name = path_splits[-3]
                    # dump video using imageio
                    os.makedirs(f"{self.test_dir}/videos", exist_ok=True)
                    imageio.mimwrite(
                        f"{self.test_dir}/videos/{iter_idx}_{scene_name}.mp4",
                        (outputs[0].cpu().numpy() * 255).astype(np.uint8),
                        format="ffmpeg",
                        fps=15,
                    )
                if self.config.render_views or self.config.render_video:
                    path_splits = tar_paths[0, 0].split("/")
                    scene_name = path_splits[-3]
                    canvas_left = rearrange(ref_imgs, "b v h w c -> (b h) (v w) c")
                    canvas = torch.cat([canvas_left], dim=0)
                    write_tensor_to_image(
                        canvas, f"{self.test_dir}/imgs/{iter_idx}_{scene_name}_reference.png"
                    )

                    canvas_right = rearrange(tar_imgs[:,::10], "b v h w c -> (b h) (v w) c")
                    canvas = torch.cat([canvas_right], dim=0)
                    write_tensor_to_image(
                        canvas, f"{self.test_dir}/imgs/{iter_idx}_{scene_name}_target.png"
                    )

                    canvas_outputs = rearrange(outputs[:,::10], "b v h w c -> (b h) (v w) c")
                    canvas = torch.cat([canvas_outputs], dim=0)
                    write_tensor_to_image(
                        canvas, f"{self.test_dir}/imgs/{iter_idx}_{scene_name}_outputs.png"
                    )
                else:
                    # dump images.
                    if len(canvas) < 10:
                        canvas_left = rearrange(ref_imgs, "b v h w c -> (b h) (v w) c")
                        canvas_right = rearrange(
                            torch.cat([tar_imgs, outputs], dim=3),
                            "b v h w c -> (b h) (v w) c",
                        )
                        canvas_mid = torch.ones(
                            len(canvas_left), 20, 3, device=self.device
                        )
                        canvas.append(
                            torch.cat([canvas_left, canvas_mid, canvas_right], dim=1)
                        )

                    # metrics.
                    outputs = rearrange(outputs, "b v h w c -> (b v) c h w")
                    tar_imgs = rearrange(tar_imgs, "b v h w c -> (b v) c h w")
                    psnrs.append(state["psnr_fn"](outputs, tar_imgs))
                    ssims.append(state["ssim_fn"](outputs, tar_imgs))
                    lpips.append(state["lpips_fn"](outputs, tar_imgs))

            if self.config.render_video:
                return

            # dump canvas.
            canvas = torch.cat(canvas, dim=0)
            write_tensor_to_image(
                canvas, f"{self.test_dir}/rank{self.world_rank}_{label}views.png"
            )

            def distributed_avg(data: List[float], name: str) -> float:
                # collect metric from all ranks
                collected_sizes = [None] * self.world_size
                torch.distributed.all_gather_object(collected_sizes, len(data))
                collected = [
                    torch.empty(size, device=self.device) for size in collected_sizes
                ]
                torch.distributed.all_gather(
                    collected, torch.tensor(data, device=self.device)
                )
                collected = torch.cat(collected)

                if torch.isinf(collected).any():
                    self.logging_on_master(
                        f"Inf found in {label} views, {sum(torch.isinf(collected))} inf values for {name}."
                    )
                    collected = collected[~torch.isinf(collected)]
                if torch.isnan(collected).any():
                    self.logging_on_master(
                        f"NaN found in {label} views, {sum(torch.isnan(collected))} nan values for {name}."
                    )
                    collected = collected[~torch.isnan(collected)]

                avg = collected.mean().item()
                return avg, len(collected)

            avg_psnr, n_total = distributed_avg(psnrs, "psnr")
            avg_lpips, n_total = distributed_avg(lpips, "lpips")
            avg_ssim, n_total = distributed_avg(ssims, "ssim")

            self.logging_on_master(
                f"PSNR{label}: {avg_psnr:.3f}, SSIM{label}: {avg_ssim:.3f}, LPIPS{label}: {avg_lpips:.3f} "
                f"evaluated on {n_total} scenes at step {step}."
            )

            if self.world_rank == 0:
                self.writer.add_scalar(f"test/psnr{label}", avg_psnr, step)
                self.writer.add_scalar(f"test/ssim{label}", avg_ssim, step)
                self.writer.add_scalar(f"test/lpips{label}", avg_lpips, step)
                with open(f"{self.test_dir}/metrics.json", "w") as f:
                    json.dump(
                        {
                            "label": label,
                            "step": step,
                            "n_total": n_total,
                            "psnr": avg_psnr,
                            "ssim": avg_ssim,
                            "lpips": avg_lpips,
                        },
                        f,
                    )


if __name__ == "__main__":
    """Example usage:

    # 2GPUs dry run
    OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
        nvs/trainval.py lvsm-dry-run --model_config.encoder.num_layers 2
    """

    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics")

    configs = {
        "lvsm": (
            "feedforward large view synthesis model",
            LVSMLauncherConfig(),
        ),
        "lvsm-dry-run": (
            "dry run",
            LVSMLauncherConfig(
                amp=True,
                amp_dtype="fp16",
                dataset_batch_scenes=1,
                max_steps=10,
                test_every=5,
                test_n=10,
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    if cfg.max_steps == 80_000 and cfg.test_every != 80_000:
        warnings.warn(
            "80k training runs evaluate only at the final step; "
            f"overriding test_every={cfg.test_every} with 80000.",
            stacklevel=2,
        )
        cfg.test_every = 80_000
    launcher = LVSMLauncher(cfg)
    launcher.run()
