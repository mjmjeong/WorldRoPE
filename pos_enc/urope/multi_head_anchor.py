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

# Portions of this file are adapted from the reference implementation of
# "PRoPE: Projective Positional Encoding for Multiview Transformers". See the
# NOTICE and THIRD_PARTY_LICENSES files for attribution and the applicable
# third-party license terms.

import math
from functools import partial
from typing import Callable, List, Optional, Tuple
import torch
import torch.nn.functional as F

from pos_enc.utils.geometry import _invert_K, _invert_SE3
from pos_enc.utils.rotary import _apply_block_diagonal, _rope_apply_coeffs, _rope_apply_mixed_coeffs, _rope_precompute_coeffs_batch, _rope_precompute_mixed_anchor_coeffs_batch, _rope_precompute_mixed_coeffs_batch
from pos_enc.worldrope.integration_probabilistic_rope import _expand_anchor_token_values, _mix_anchor_mixed_rope_coeffs, _mix_anchor_rope_coeffs, _rope_precompute_gaussian_coeffs_batch, _rope_precompute_mixed_gaussian_anchor_mixture_coeffs_batch, _rope_precompute_mixed_gaussian_coeffs_batch
from pos_enc.worldrope.token_with_gaussian_splatting import _project_fixed_lateral_footprint_covariance, _urope_image_coordinate_and_depth_derivative


############################################################################
# Prepare camera-dependent Q, K/V and output transforms for this
# positional-encoding variant.
############################################################################
def _prepare_apply_fns(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    group_size: int,
    depth_num: int,
    max_depth: float,
    min_depth: float,
    anchor_weights: Optional[torch.Tensor] = None,
    anchor_sigmas: Optional[torch.Tensor] = None,
    anchor_means: Optional[torch.Tensor] = None,
    fixed_lateral_footprint_kv: bool = False,
    mixed_freqs: Optional[torch.Tensor] = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for URoPE-style positional encoding."""
    # viewmats is `camera<-world` transform (x-right, y-up, z-backward).

    device = viewmats.device
    coordinate_transforms = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=viewmats.dtype,
    )
    viewmats = coordinate_transforms[None, None] @ viewmats # (batch, cameras, 4, 4)
    num_cameras = viewmats.shape[1]
    # now viewmats is `camera<-world` transform (x-right, y-down, z-forward).

    (batch, cameras, _, _) = viewmats.shape
    Ks_inv = _invert_K(Ks)  # (batch, cameras, 3, 3)
    viewmats_cam2world = _invert_SE3(viewmats)  # (batch, cameras, 4, 4)

    grid_x, grid_y = torch.meshgrid(torch.arange(patches_x), torch.arange(patches_y), indexing="xy")
    img_x = (grid_x + 0.5) * (image_width / patches_x)
    img_y = (grid_y + 0.5) * (image_height / patches_y)
    img_z = torch.ones_like(img_x)
    img_xyz = torch.stack([img_x, img_y, img_z], dim=-1).to(device)  # (patch_y, patch_x, 3)
    img_xyz = img_xyz.reshape(-1, 3)[None, None].repeat(batch, cameras, 1, 1)  # (batch, cameras, patch_num, 3)
    cam_xyz = img_xyz @ Ks_inv.transpose(-1, -2) # (batch, cameras, patch_num, 3)
    cam_xyz_pad = torch.cat([cam_xyz, torch.ones_like(cam_xyz[..., :1])], dim=-1) # (batch, cameras, patch_num, 4)

    # get the ray map in the camera coordinate
    raymap = cam_xyz_pad @ viewmats_cam2world.transpose(-1, -2)  # (batch, cameras, patch_num, 4)
    raydir = raymap[..., :3] - viewmats_cam2world[..., None, :3, 3]

    P = viewmats_cam2world[:, :, None].repeat(1, 1, patches_x * patches_y, 1, 1)

    cam_centers = P[..., :3, 3]  # (batch, cameras, patch_num, 3) world coordinates
    ray_ends = cam_centers + raydir  # (batch, cameras, patch_num, 3) world coordinates

    cam2world = viewmats_cam2world   # (batch, cameras, 4, 4)

    cam2cam = _invert_SE3(cam2world)[:, None] @ cam2world[:, :, None]  # (batch, cameras, cameras, 4, 4)  1 --> 2

    cam_centers_pad = torch.cat([cam_centers, torch.ones_like(cam_centers[..., :1])], dim=-1)  # (batch, cameras, patch_num, 4)
    ray_ends_pad = torch.cat([ray_ends, torch.ones_like(ray_ends[..., :1])], dim=-1)  # (batch, cameras, patch_num, 4)

    cam_centers_pad = cam_centers_pad @ viewmats.transpose(-1, -2) # (batch, cameras, patch_num, 4) camera coordinates
    ray_ends_pad = ray_ends_pad @ viewmats.transpose(-1, -2) # (batch, cameras, patch_num, 4) camera coordinates

    # convert the ray map into the query camera coordinate
    cam_centers_camcoord = cam_centers_pad[:, :, None] @ cam2cam.transpose(-1, -2)  # (batch, cameras, cameras, patch_num, 4)  each cam_1 in cam_2 coordinates
    ray_ends_camcoord = ray_ends_pad[:, :, None] @ cam2cam.transpose(-1, -2)  # (batch, cameras, cameras, patch_num, 4)

    cam_centers_camcoord = cam_centers_camcoord[..., :3]  # (batch, cameras, cameras, patch_num, 3)
    ray_ends_camcoord = ray_ends_camcoord[..., :3]  # (batch, cameras, cameras, patch_num, 3)
    ray_dirs_camcoord = ray_ends_camcoord - cam_centers_camcoord  # (batch, cameras, cameras, patch_num, 3)

    cam_ids = torch.arange(cameras, device=device)[None, :, None, None, None].repeat(
        batch, 1, 1, patches_x * patches_y, 3
    )  # [batch, cameras, 1, patch_num, 3]
    q_cam_centers_camcoord = torch.gather(cam_centers_camcoord, dim=2, index=cam_ids)  # [batch, cameras, 1, patch_num, 3]
    q_ray_dirs_camcoord = torch.gather(ray_dirs_camcoord, dim=2, index=cam_ids)  # [batch, cameras, 1, patch_num, 3]

    q_cam_centers_camcoord = q_cam_centers_camcoord.squeeze(2)  # [batch, cameras, patch_num, 3]
    q_ray_dirs_camcoord = q_ray_dirs_camcoord.squeeze(2)  # [batch, cameras, patch_num, 3]


    q_ray_x = q_ray_dirs_camcoord[..., 0].flatten(0, 1)  # [batch*cameras, patch_num]
    q_ray_y = q_ray_dirs_camcoord[..., 1].flatten(0, 1)  # [batch*cameras, patch_num]
    q_ray_z = q_ray_dirs_camcoord[..., 2].flatten(0, 1)  # [batch*cameras, patch_num]

    q_cam_x = q_cam_centers_camcoord[..., 0].flatten(0, 1)  # [batch*cameras, patch_num]
    q_cam_y = q_cam_centers_camcoord[..., 1].flatten(0, 1)  # [batch*cameras, patch_num]
    q_cam_z = q_cam_centers_camcoord[..., 2].flatten(0, 1)  # [batch*cameras, patch_num]


    # sample the depth anchors along the ray
    index = torch.arange(start=0, end=depth_num, step=1, device=device).float()
    bin_size = (max_depth - min_depth) / depth_num
    depths = min_depth + bin_size * index
    depths = depths[None, :, None]  # [1, depth_num, 1]

    q_x = q_cam_x[:, None] + q_ray_x[:, None] * depths  # [batch*cameras, depth_num, patch_num]
    q_y = q_cam_y[:, None] + q_ray_y[:, None] * depths  # [batch*cameras, depth_num, patch_num]
    q_z = q_cam_z[:, None] + q_ray_z[:, None] * depths  # [batch*cameras, depth_num, patch_num]

    q_x = q_x.repeat(1, group_size, 1)  # [batch*cameras, num_heads, patch_num]
    q_y = q_y.repeat(1, group_size, 1)  # [batch*cameras, num_heads, patch_num]
    q_z = q_z.repeat(1, group_size, 1)  # [batch*cameras, num_heads, patch_num]

    q_img_x = img_x / (image_width / patches_x)
    q_img_y = img_y / (image_height / patches_y)
    q_img_x = q_img_x.flatten(0, 1)[None, None].repeat(batch*cameras, depth_num*group_size, 1)  # [batch*cameras, num_heads, patch_num]
    q_img_y = q_img_y.flatten(0, 1)[None, None].repeat(batch*cameras, depth_num*group_size, 1)  # [batch*cameras, num_heads, patch_num]
    q_img_x = q_img_x.type_as(q_x)
    q_img_y = q_img_y.type_as(q_y)

    if mixed_freqs is None:
        q_coeffs_img_x = _rope_precompute_coeffs_batch(
            q_img_x, 100, 1, head_dim // 2
        )
        q_coeffs_img_y = _rope_precompute_coeffs_batch(
            q_img_y, 100, 1, head_dim // 2
        )
    else:
        q_coeffs_mixed = _rope_precompute_mixed_coeffs_batch(
            q_img_x, q_img_y, mixed_freqs
        )


    cam_centers_camcoord = cam_centers_camcoord.permute(0, 2, 1, 3, 4)
    ray_dirs_camcoord = ray_dirs_camcoord.permute(0, 2, 1, 3, 4)
    kv_cam_x = cam_centers_camcoord[..., 0].reshape(batch*num_cameras, num_cameras*patches_x * patches_y) # [batch*cameras, cameras*patch_num]
    kv_cam_y = cam_centers_camcoord[..., 1].reshape(batch*num_cameras, num_cameras*patches_x * patches_y) # [batch*cameras, cameras*patch_num]
    kv_cam_z = cam_centers_camcoord[..., 2].reshape(batch*num_cameras, num_cameras*patches_x * patches_y) # [batch*cameras, cameras*patch_num]

    kv_ray_x = ray_dirs_camcoord[..., 0].reshape(batch*num_cameras, num_cameras*patches_x * patches_y) # [batch*cameras, cameras*patch_num]
    kv_ray_y = ray_dirs_camcoord[..., 1].reshape(batch*num_cameras, num_cameras*patches_x * patches_y) # [batch*cameras, cameras*patch_num]
    kv_ray_z = ray_dirs_camcoord[..., 2].reshape(batch*num_cameras, num_cameras*patches_x * patches_y) # [batch*cameras, cameras*patch_num]


    kv_depths = depths
    if anchor_means is not None:
        assert anchor_weights is not None and anchor_sigmas is not None
        assert anchor_means.shape == anchor_weights.shape
        # [B,T,D] -> [B*Q,D,T].  Means belong to source K/V tokens and are
        # shared across query cameras; only their projection is query-specific.
        kv_depths = _expand_anchor_token_values(
            anchor_means,
            num_cameras=num_cameras,
        ).to(dtype=kv_cam_x.dtype)

    kv_x = kv_cam_x[:, None] + kv_ray_x[:, None] * kv_depths  # [batch*cameras, depth_num, cameras*patch_num]
    kv_y = kv_cam_y[:, None] + kv_ray_y[:, None] * kv_depths  # [batch*cameras, depth_num, cameras*patch_num]
    kv_z = kv_cam_z[:, None] + kv_ray_z[:, None] * kv_depths  # [batch*cameras, depth_num, cameras*patch_num]

    kv_xyz = torch.stack([kv_x, kv_y, kv_z], dim=-1) # [batch*cameras, depth_num, cameras*patch_num, 3]
    query_Ks = Ks.flatten(0, 1)
    kv_img_xyz = kv_xyz @ query_Ks[:, None].transpose(-1, -2) # [batch*cameras, depth_num, cameras*patch_num, 3]

    kv_img_x_unclamped, derivative_x = (
        _urope_image_coordinate_and_depth_derivative(
            kv_img_xyz,
            projected_depth_derivative=None,
            coordinate_index=0,
            coordinate_scale=(image_width / patches_x),
        )
    )
    kv_img_y_unclamped, derivative_y = (
        _urope_image_coordinate_and_depth_derivative(
            kv_img_xyz,
            projected_depth_derivative=None,
            coordinate_index=1,
            coordinate_scale=(image_height / patches_y),
        )
    )

    gaussian_variance_x = gaussian_variance_y = None
    gaussian_covariance_xy = None
    if fixed_lateral_footprint_kv:
        (
            gaussian_variance_x,
            gaussian_variance_y,
            gaussian_covariance_xy,
        ) = _project_fixed_lateral_footprint_covariance(
            component_depths=kv_depths,
            projected_component_means=kv_img_xyz,
            query_Ks=query_Ks,
            source_Ks_inv=Ks_inv,
            camera_to_camera=cam2cam,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_width,
            image_height=image_height,
        )

    if anchor_sigmas is not None:
        assert anchor_weights is not None
        assert anchor_sigmas.shape == anchor_weights.shape

        # For every fixed component mean D_i, linearize URoPE's exact
        # key-to-query image projection g(D) along the source key ray.  This is
        # the rank-one depth-only Gaussian used by RayRoPE, expressed in
        # URoPE's two image-plane coordinate axes.
        kv_ray_xyz = torch.stack([kv_ray_x, kv_ray_y, kv_ray_z], dim=-1)
        projected_depth_derivative = (
            kv_ray_xyz @ query_Ks.transpose(-1, -2)
        )[:, None]
        kv_img_x_unclamped, derivative_x = (
            _urope_image_coordinate_and_depth_derivative(
                kv_img_xyz,
                projected_depth_derivative=projected_depth_derivative,
                coordinate_index=0,
                coordinate_scale=(image_width / patches_x),
            )
        )
        kv_img_y_unclamped, derivative_y = (
            _urope_image_coordinate_and_depth_derivative(
                kv_img_xyz,
                projected_depth_derivative=projected_depth_derivative,
                coordinate_index=1,
                coordinate_scale=(image_height / patches_y),
            )
        )
        # Sigma is intentionally unrestricted.  Keep its projection and
        # variance calculation in FP32 under the launcher's FP16 autocast so
        # this design choice does not accidentally inherit FP16's much smaller
        # finite range.
        derivative_x = derivative_x.float()
        derivative_y = derivative_y.float()

        # The deterministic URoPE baseline clamps projected image positions.
        # Use the corresponding first-order derivative: saturated coordinates
        # have zero local variance, while in-range coordinates retain the
        # Gaussian depth uncertainty.
        derivative_x = derivative_x * (
            (kv_img_x_unclamped > -64) & (kv_img_x_unclamped < 96)
        ).to(derivative_x.dtype)
        derivative_y = derivative_y * (
            (kv_img_y_unclamped > -64) & (kv_img_y_unclamped < 96)
        ).to(derivative_y.dtype)

        expanded_anchor_sigmas = _expand_anchor_token_values(
            anchor_sigmas,
            num_cameras=num_cameras,
        ).to(dtype=torch.float32)
        depth_std_x = derivative_x * expanded_anchor_sigmas
        depth_std_y = derivative_y * expanded_anchor_sigmas
        depth_variance_x = depth_std_x.square()
        depth_variance_y = depth_std_y.square()
        depth_covariance_xy = depth_std_x * depth_std_y
        if gaussian_variance_x is None:
            gaussian_variance_x = depth_variance_x
            gaussian_variance_y = depth_variance_y
            gaussian_covariance_xy = depth_covariance_xy
        else:
            # Direct-3D footprint: depth and the two fixed lateral variables
            # are independent, so their projected covariances add.  In
            # particular, no sigma_D^2 * sigma_UV^2 cross term is introduced.
            gaussian_variance_x = gaussian_variance_x + depth_variance_x
            gaussian_variance_y = gaussian_variance_y + depth_variance_y
            gaussian_covariance_xy = (
                gaussian_covariance_xy + depth_covariance_xy
            )

    kv_img_x = torch.clamp(kv_img_x_unclamped, min=-64, max=96)
    kv_img_y = torch.clamp(kv_img_y_unclamped, min=-64, max=96)
    has_kv_gaussian = gaussian_variance_x is not None
    if has_kv_gaussian:
        assert gaussian_variance_y is not None
        assert gaussian_covariance_xy is not None

    if mixed_freqs is None:
        if anchor_weights is None:
            kv_img_x = kv_img_x.repeat_interleave(group_size, dim=1)
            kv_img_y = kv_img_y.repeat_interleave(group_size, dim=1)
            if has_kv_gaussian:
                repeated_variance_x = gaussian_variance_x.repeat_interleave(
                    group_size, dim=1
                )
                repeated_variance_y = gaussian_variance_y.repeat_interleave(
                    group_size, dim=1
                )
                kv_coeffs_img_x = _rope_precompute_gaussian_coeffs_batch(
                    kv_img_x,
                    repeated_variance_x,
                    100,
                    1,
                    head_dim // 2,
                )
                kv_coeffs_img_y = _rope_precompute_gaussian_coeffs_batch(
                    kv_img_y,
                    repeated_variance_y,
                    100,
                    1,
                    head_dim // 2,
                )
            else:
                kv_coeffs_img_x = _rope_precompute_coeffs_batch(
                    kv_img_x, 100, 1, head_dim // 2
                )
                kv_coeffs_img_y = _rope_precompute_coeffs_batch(
                    kv_img_y, 100, 1, head_dim // 2
                )
        else:
            # Predict one four-anchor distribution per source token and layer.
            assert anchor_weights.shape == (
                batch,
                num_cameras * patches_x * patches_y,
                depth_num,
            )
            if not has_kv_gaussian:
                kv_coeffs_img_x_per_anchor = _rope_precompute_coeffs_batch(
                    kv_img_x, 100, 1, head_dim // 2
                )
                kv_coeffs_img_y_per_anchor = _rope_precompute_coeffs_batch(
                    kv_img_y, 100, 1, head_dim // 2
                )
            else:
                kv_coeffs_img_x_per_anchor = _rope_precompute_gaussian_coeffs_batch(
                    kv_img_x,
                    gaussian_variance_x,
                    100,
                    1,
                    head_dim // 2,
                )
                kv_coeffs_img_y_per_anchor = _rope_precompute_gaussian_coeffs_batch(
                    kv_img_y,
                    gaussian_variance_y,
                    100,
                    1,
                    head_dim // 2,
                )
            kv_coeffs_img_x = _mix_anchor_rope_coeffs(
                kv_coeffs_img_x_per_anchor,
                anchor_weights,
                num_cameras=num_cameras,
            )
            kv_coeffs_img_y = _mix_anchor_rope_coeffs(
                kv_coeffs_img_y_per_anchor,
                anchor_weights,
                num_cameras=num_cameras,
            )

        transforms_q = [
            (partial(_rope_apply_coeffs, coeffs=q_coeffs_img_x), head_dim // 2),
            (partial(_rope_apply_coeffs, coeffs=q_coeffs_img_y), head_dim // 2),
        ]
        transforms_kv = [
            (partial(_rope_apply_coeffs, coeffs=kv_coeffs_img_x), head_dim // 2),
            (partial(_rope_apply_coeffs, coeffs=kv_coeffs_img_y), head_dim // 2),
        ]
        transforms_o = [
            (
                partial(_rope_apply_coeffs, coeffs=q_coeffs_img_x, inverse=True),
                head_dim // 2,
            ),
            (
                partial(_rope_apply_coeffs, coeffs=q_coeffs_img_y, inverse=True),
                head_dim // 2,
            ),
        ]
    else:
        if anchor_weights is None:
            kv_img_x = kv_img_x.repeat_interleave(group_size, dim=1)
            kv_img_y = kv_img_y.repeat_interleave(group_size, dim=1)
            if has_kv_gaussian:
                kv_coeffs_mixed = _rope_precompute_mixed_gaussian_coeffs_batch(
                    positions_x=kv_img_x,
                    positions_y=kv_img_y,
                    variance_x=gaussian_variance_x.repeat_interleave(
                        group_size, dim=1
                    ),
                    variance_y=gaussian_variance_y.repeat_interleave(
                        group_size, dim=1
                    ),
                    covariance_xy=gaussian_covariance_xy.repeat_interleave(
                        group_size, dim=1
                    ),
                    frequencies=mixed_freqs,
                )
            else:
                kv_coeffs_mixed = _rope_precompute_mixed_coeffs_batch(
                    kv_img_x, kv_img_y, mixed_freqs
                )
        else:
            assert anchor_weights.shape == (
                batch,
                num_cameras * patches_x * patches_y,
                depth_num,
            )
            if fixed_lateral_footprint_kv:
                kv_coeffs_mixed = (
                    _rope_precompute_mixed_gaussian_anchor_mixture_coeffs_batch(
                        positions_x=kv_img_x,
                        positions_y=kv_img_y,
                        variance_x=gaussian_variance_x,
                        variance_y=gaussian_variance_y,
                        covariance_xy=gaussian_covariance_xy,
                        frequencies=mixed_freqs,
                        anchor_weights=anchor_weights,
                        num_cameras=num_cameras,
                    )
                )
            else:
                kv_coeffs_per_anchor = _rope_precompute_mixed_anchor_coeffs_batch(
                    positions_x=kv_img_x,
                    positions_y=kv_img_y,
                    derivative_x=(
                        derivative_x if anchor_sigmas is not None else None
                    ),
                    derivative_y=(
                        derivative_y if anchor_sigmas is not None else None
                    ),
                    sigmas=(
                        expanded_anchor_sigmas
                        if anchor_sigmas is not None
                        else None
                    ),
                    frequencies=mixed_freqs,
                )
                kv_coeffs_mixed = _mix_anchor_mixed_rope_coeffs(
                    kv_coeffs_per_anchor,
                    anchor_weights,
                    num_cameras=num_cameras,
                )

        num_base_freqs = head_dim // 4
        q_coeffs_first = tuple(
            coeff[..., :num_base_freqs] for coeff in q_coeffs_mixed
        )
        q_coeffs_second = tuple(
            coeff[..., num_base_freqs:] for coeff in q_coeffs_mixed
        )
        kv_coeffs_first = tuple(
            coeff[..., :num_base_freqs] for coeff in kv_coeffs_mixed
        )
        kv_coeffs_second = tuple(
            coeff[..., num_base_freqs:] for coeff in kv_coeffs_mixed
        )
        transforms_q = [
            (partial(_rope_apply_mixed_coeffs, coeffs=q_coeffs_first), head_dim // 2),
            (partial(_rope_apply_mixed_coeffs, coeffs=q_coeffs_second), head_dim // 2),
        ]
        transforms_kv = [
            (partial(_rope_apply_mixed_coeffs, coeffs=kv_coeffs_first), head_dim // 2),
            (partial(_rope_apply_mixed_coeffs, coeffs=kv_coeffs_second), head_dim // 2),
        ]
        transforms_o = [
            (
                partial(
                    _rope_apply_mixed_coeffs,
                    coeffs=q_coeffs_first,
                    inverse=True,
                ),
                head_dim // 2,
            ),
            (
                partial(
                    _rope_apply_mixed_coeffs,
                    coeffs=q_coeffs_second,
                    inverse=True,
                ),
                head_dim // 2,
            ),
        ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o
