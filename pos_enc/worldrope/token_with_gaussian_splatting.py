"""Predict token-wise Gaussian supports and project their geometry across views.

The term splatting refers to finite Gaussian support, not rasterization.
No token features are rasterized or warped. Expected rotary integration lives
in integration_probabilistic_rope.py; shared QKV/SDPA lives in pos_enc.utils.
"""

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
from pos_enc.utils.rotary import _apply_block_diagonal, _prepare_explicit_axial_coeffs, _rope_apply_mixed_coeffs, _rope_precompute_mixed_coeffs_batch
from pos_enc.worldrope.integration_probabilistic_rope import _prepare_gmm_axial_factor_coeffs, _prepare_gmm_mixed_factor_coeffs


from torch.nn import Parameter

############################################################################
# Register per-token Gaussian mean, log-sigma and mixture-logit heads.
# Zero weights and fixed mean biases preserve the original initialization.
# Parameters stay on the owning MHA module to retain checkpoint tensor keys.
############################################################################
def initialize_gmm_heads(self, embed_dim, predict_anchor_weights, predict_anchor_sigmas, predict_anchor_means, anchor_mean_init):
    if predict_anchor_weights > 0:
        self.anchor_proj_weight = Parameter(
            torch.zeros((predict_anchor_weights, embed_dim))
        )
        self.anchor_proj_bias = Parameter(torch.zeros(predict_anchor_weights))
    else:
        self.register_parameter("anchor_proj_weight", None)
        self.register_parameter("anchor_proj_bias", None)

    # Optional additive metric-depth standard deviation for every fixed
    # URoPE anchor.  The raw outputs are unconstrained log-sigmas; an
    # unclamped exp below gives sigma>0.  Zero initialization therefore
    # starts every component at sigma=1 without consuming init RNG.
    if predict_anchor_sigmas > 0:
        if predict_anchor_sigmas != predict_anchor_weights:
            raise ValueError(
                "Anchor sigma and mixture-weight heads must have the same "
                "number of components."
            )
        self.anchor_sigma_proj_weight = Parameter(
            torch.zeros((predict_anchor_sigmas, embed_dim))
        )
        self.anchor_sigma_proj_bias = Parameter(
            torch.zeros(predict_anchor_sigmas)
        )
    else:
        self.register_parameter("anchor_sigma_proj_weight", None)
        self.register_parameter("anchor_sigma_proj_bias", None)

    # Optional learnable metric-depth mean for each GMM component.  This
    # is another pointwise pre-QKV head.  Its zero weight and fixed-anchor
    # bias make the step-zero forward exactly equal to the fixed-mean GMM.
    if predict_anchor_means > 0:
        if predict_anchor_means != predict_anchor_weights:
            raise ValueError(
                "Anchor mean and mixture-weight heads must have the same "
                "number of components."
            )
        if anchor_mean_init is None or len(anchor_mean_init) != predict_anchor_means:
            raise ValueError(
                "anchor_mean_init must provide one initial mean per component."
            )
        self.anchor_mean_proj_weight = Parameter(
            torch.zeros((predict_anchor_means, embed_dim))
        )
        self.anchor_mean_proj_bias = Parameter(
            torch.tensor(tuple(anchor_mean_init), dtype=torch.float32)
        )
    else:
        self.register_parameter("anchor_mean_proj_weight", None)
        self.register_parameter("anchor_mean_proj_bias", None)


############################################################################
# Predict Gaussian parameters from the pre-QKV token features [B,T,C].
# Return weights, sigmas and means [B,T,G], shared across attention heads.
# Use FP32 softmax/exp; means are nonnegative and sigma has no upper clamp.
############################################################################
def predict_gmm_parameters(self, query):
    anchor_weights = None
    if self.predict_anchor_weights > 0:
        anchor_logits = F.linear(
            query, self.anchor_proj_weight, self.anchor_proj_bias
        )
        # Keep the four-way simplex calculation numerically stable under
        # the model's FP16 autocast context.
        anchor_weights = F.softmax(anchor_logits.float(), dim=-1)
    anchor_sigmas = None
    if self.predict_anchor_sigmas > 0:
        anchor_log_sigmas = F.linear(
            query,
            self.anchor_sigma_proj_weight,
            self.anchor_sigma_proj_bias,
        )
        # Deliberately no clamp: this experiment learns an unrestricted
        # positive additive depth sigma for each Gaussian component.
        anchor_sigmas = torch.exp(anchor_log_sigmas.float())
    anchor_means = None
    if self.predict_anchor_means > 0:
        raw_anchor_means = F.linear(
            query,
            self.anchor_mean_proj_weight,
            self.anchor_mean_proj_bias,
        )
        # The requested constraint is on the component mean only.  Sigma
        # remains unrestricted and independent of this hard lower bound.
        anchor_means = torch.clamp(raw_anchor_means.float(), min=0.0)

    return anchor_weights, anchor_sigmas, anchor_means


############################################################################
# Lift Gaussian token supports into 3D and project them into every query view.
# Inputs use Blender world-to-camera poses, pixel K, and per-token GMM values.
# Carry covariance as factors; delegate expected rotations to the integration module.
# Return Q, K/V and output transforms in patch-grid U/V and signed-depth units.
############################################################################
def _prepare_apply_fns_uvd(
    head_dim: int,
    viewmats: torch.Tensor,
    Ks: Optional[torch.Tensor],
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    group_size: int,
    depth_num: int,
    max_depth: float,
    min_depth: float,
    anchor_weights: Optional[torch.Tensor] = None,
    anchor_sigmas: Optional[torch.Tensor] = None,
    anchor_means: Optional[torch.Tensor] = None,
    fixed_lateral_footprint_kv: bool = False,
    mixed_freqs: Optional[torch.Tensor] = None,
    coordinate_mode: str = "uvd_axial",
    uv_axial_freqs: Optional[torch.Tensor] = None,
    depth_axial_freqs: Optional[torch.Tensor] = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare the isolated signed-depth UVD Gaussian-RoPE variants.

    This function is never called by legacy ``urope`` or ``urope_mixed``.
    It constructs each source token's Direct-3D Gaussian in the source camera,
    rigidly transforms it to every query camera, and applies one pinhole
    Jacobian to obtain the full covariance of (u, v, signed query z).
    """
    del group_size  # UVD modes apply every coordinate in every attention head.
    if coordinate_mode not in {
        "uvd_axial",
        "uv_mixed_d_axial",
        "uvd_mixed",
    }:
        raise ValueError(f"Unexpected UVD coordinate mode {coordinate_mode!r}.")
    if Ks is None:
        raise ValueError("UVD URoPE requires camera intrinsics.")
    if anchor_weights is None or anchor_sigmas is None:
        raise ValueError("UVD URoPE requires learned GMM weights and sigmas.")
    if uv_axial_freqs is None or depth_axial_freqs is None:
        raise ValueError("UVD URoPE requires explicit U/V and depth frequencies.")
    if head_dim % 6 != 0:
        raise ValueError(f"Expected head_dim divisible by 6, got {head_dim}.")
    if not fixed_lateral_footprint_kv:
        raise ValueError(
            "These UVD experiments are defined on the fixed-lateral GMM+splat base."
        )

    device_type = viewmats.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        viewmats_f = viewmats.float()
        Ks_f = Ks.float()
        coordinate_transform = viewmats_f.new_tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        # Input camera<-world matrices are Blender/OpenGL.  Convert once to
        # x-right, y-down, z-forward before forming source-to-query transforms.
        w2c = coordinate_transform[None, None] @ viewmats_f
        c2w = _invert_SE3(w2c)
        batch, cameras = w2c.shape[:2]
        patch_num = patches_x * patches_y
        tokens = cameras * patch_num
        components = depth_num
        if anchor_weights.shape != (batch, tokens, components):
            raise ValueError(f"Unexpected GMM weight shape {anchor_weights.shape}.")
        if anchor_sigmas.shape != anchor_weights.shape:
            raise ValueError(f"Unexpected GMM sigma shape {anchor_sigmas.shape}.")
        if anchor_means is not None and anchor_means.shape != anchor_weights.shape:
            raise ValueError(f"Unexpected GMM mean shape {anchor_means.shape}.")

        # Patch-center rays in each source camera.  K^{-1}[x,y,1] has z=1,
        # so learned component means remain optical-axis depths.
        grid_x, grid_y = torch.meshgrid(
            torch.arange(patches_x, device=viewmats.device, dtype=torch.float32),
            torch.arange(patches_y, device=viewmats.device, dtype=torch.float32),
            indexing="xy",
        )
        pixel_x = (grid_x + 0.5) * (image_width / patches_x)
        pixel_y = (grid_y + 0.5) * (image_height / patches_y)
        pixel_h = torch.stack([pixel_x, pixel_y, torch.ones_like(pixel_x)], dim=-1)
        pixel_h = pixel_h.reshape(1, 1, patch_num, 3).expand(
            batch, cameras, patch_num, 3
        )
        Ks_inv = _invert_K(Ks_f)
        source_rays = pixel_h @ Ks_inv.transpose(-1, -2)

        # [B,Q,S,4,4], exactly source-S camera coordinates -> query-Q.
        source_to_query = w2c[:, :, None] @ c2w[:, None, :]
        rotations = source_to_query[..., :3, :3]
        translations = source_to_query[..., :3, 3]
        query_rays = torch.einsum("bqsij,bspj->bqspi", rotations, source_rays)

        if anchor_means is None:
            anchor_index = torch.arange(
                components, device=viewmats.device, dtype=torch.float32
            )
            bin_size = (max_depth - min_depth) / components
            fixed_means = min_depth + bin_size * anchor_index
            source_means = fixed_means.reshape(1, 1, 1, components).expand(
                batch, cameras, patch_num, components
            )
        else:
            source_means = anchor_means.float().reshape(
                batch, cameras, patch_num, components
            )
        source_sigmas = anchor_sigmas.float().reshape(
            batch, cameras, patch_num, components
        )
        source_weights = anchor_weights.float().reshape(
            batch, cameras, patch_num, components
        )

        # Query-token GMM: U/V are deterministic patch centers and only signed
        # optical-axis depth is uncertain.  Q/O deliberately have no lateral
        # footprint in this ablation family.
        q_means = source_means.permute(0, 1, 3, 2).reshape(
            batch * cameras, components, patch_num
        )
        q_sigmas = source_sigmas.permute(0, 1, 3, 2).reshape(
            batch * cameras, components, patch_num
        )
        q_weights = source_weights.permute(0, 1, 3, 2).reshape(
            batch * cameras, components, patch_num
        )
        q_u = (grid_x + 0.5).reshape(1, patch_num).expand(
            batch * cameras, patch_num
        )
        q_v = (grid_y + 0.5).reshape(1, patch_num).expand(
            batch * cameras, patch_num
        )
        q_positions = torch.stack(
            [
                q_u[:, None].expand(-1, components, -1),
                q_v[:, None].expand(-1, components, -1),
                q_means,
            ],
            dim=-1,
        )
        q_factors = torch.zeros(
            q_positions.shape[:-1] + (1, 3),
            dtype=torch.float32,
            device=viewmats.device,
        )
        q_factors[..., 0, 2] = q_sigmas

        # Broadcast source-token parameters over all query cameras and place
        # each component mean on its transformed source ray.
        kv_means = (
            source_means.permute(0, 3, 1, 2)
            .reshape(batch, components, tokens)[:, None]
            .expand(batch, cameras, components, tokens)
            .reshape(batch * cameras, components, tokens)
        )
        kv_sigmas = (
            source_sigmas.permute(0, 3, 1, 2)
            .reshape(batch, components, tokens)[:, None]
            .expand(batch, cameras, components, tokens)
            .reshape(batch * cameras, components, tokens)
        )
        kv_weights = (
            source_weights.permute(0, 3, 1, 2)
            .reshape(batch, components, tokens)[:, None]
            .expand(batch, cameras, components, tokens)
            .reshape(batch * cameras, components, tokens)
        )
        query_rays_flat = query_rays.reshape(batch * cameras, tokens, 3)
        translations_flat = (
            translations[:, :, :, None, :]
            .expand(batch, cameras, cameras, patch_num, 3)
            .reshape(batch * cameras, tokens, 3)
        )
        kv_xyz = (
            translations_flat[:, None]
            + query_rays_flat[:, None] * kv_means[..., None]
        )

        # Three independent Gaussian factors in query-camera XYZ: learned
        # depth, fixed source-patch U, and fixed source-patch V.
        depth_factor = query_rays_flat[:, None] * kv_sigmas[..., None]
        source_u_tangent = Ks_inv[..., :, 0]
        source_v_tangent = Ks_inv[..., :, 1]
        query_u_tangent = torch.einsum(
            "bqsij,bsj->bqsi", rotations, source_u_tangent
        )
        query_v_tangent = torch.einsum(
            "bqsij,bsj->bqsi", rotations, source_v_tangent
        )
        query_u_tangent = (
            query_u_tangent[:, :, :, None]
            .expand(batch, cameras, cameras, patch_num, 3)
            .reshape(batch * cameras, tokens, 3)
        )
        query_v_tangent = (
            query_v_tangent[:, :, :, None]
            .expand(batch, cameras, cameras, patch_num, 3)
            .reshape(batch * cameras, tokens, 3)
        )
        sigma_u_pixels = (image_width / patches_x) / math.sqrt(12.0)
        sigma_v_pixels = (image_height / patches_y) / math.sqrt(12.0)
        lateral_u_factor = (
            query_u_tangent[:, None]
            * kv_means[..., None]
            * sigma_u_pixels
        )
        lateral_v_factor = (
            query_v_tangent[:, None]
            * kv_means[..., None]
            * sigma_v_pixels
        )

        # One Jacobian of (x,y,z)_query -> (u_patch,v_patch,signed z_query).
        query_Ks = Ks_f[:, :, None].expand(
            batch, cameras, cameras, 3, 3
        ).reshape(batch * cameras, cameras, 3, 3)
        query_Ks = (
            query_Ks[:, :, None]
            .expand(batch * cameras, cameras, patch_num, 3, 3)
            .reshape(batch * cameras, tokens, 3, 3)
        )
        projected = torch.einsum("btij,bdtj->bdti", query_Ks, kv_xyz)
        projected_z = projected[..., 2]
        denominator = projected_z.abs() + 1e-5
        u_unclamped = projected[..., 0] / denominator / (image_width / patches_x)
        v_unclamped = projected[..., 1] / denominator / (image_height / patches_y)
        denominator_gradient = (
            torch.sign(projected_z)[..., None] * query_Ks[:, None, :, 2, :]
        )
        grad_u = (
            query_Ks[:, None, :, 0, :] * denominator[..., None]
            - projected[..., 0, None] * denominator_gradient
        ) / denominator.square()[..., None] / (image_width / patches_x)
        grad_v = (
            query_Ks[:, None, :, 1, :] * denominator[..., None]
            - projected[..., 1, None] * denominator_gradient
        ) / denominator.square()[..., None] / (image_height / patches_y)
        u_active = ((u_unclamped > -64.0) & (u_unclamped < 96.0)).float()
        v_active = ((v_unclamped > -64.0) & (v_unclamped < 96.0)).float()
        grad_u = grad_u * u_active[..., None]
        grad_v = grad_v * v_active[..., None]
        grad_z = torch.zeros_like(grad_u)
        grad_z[..., 2] = 1.0
        jacobian = torch.stack([grad_u, grad_v, grad_z], dim=-2)
        xyz_factors = torch.stack(
            [depth_factor, lateral_u_factor, lateral_v_factor], dim=-2
        )
        uvd_factors = torch.einsum(
            "bdtoj,bdtlj->bdtlo", jacobian, xyz_factors
        )
        kv_positions = torch.stack(
            [
                u_unclamped.clamp(-64.0, 96.0),
                v_unclamped.clamp(-64.0, 96.0),
                kv_xyz[..., 2],
            ],
            dim=-1,
        )

    block_dim = head_dim // 3
    if coordinate_mode == "uvd_axial":
        q_coeffs_u = _prepare_explicit_axial_coeffs(q_u[:, None], uv_axial_freqs)
        q_coeffs_v = _prepare_explicit_axial_coeffs(q_v[:, None], uv_axial_freqs)
        q_coeffs_d = _prepare_gmm_axial_factor_coeffs(
            q_positions[..., 2],
            q_factors[..., 2],
            q_weights,
            depth_axial_freqs,
        )
        kv_coeffs_u = _prepare_gmm_axial_factor_coeffs(
            kv_positions[..., 0],
            uvd_factors[..., 0],
            kv_weights,
            uv_axial_freqs,
        )
        kv_coeffs_v = _prepare_gmm_axial_factor_coeffs(
            kv_positions[..., 1],
            uvd_factors[..., 1],
            kv_weights,
            uv_axial_freqs,
        )
        kv_coeffs_d = _prepare_gmm_axial_factor_coeffs(
            kv_positions[..., 2],
            uvd_factors[..., 2],
            kv_weights,
            depth_axial_freqs,
        )
        q_blocks = [q_coeffs_u, q_coeffs_v, q_coeffs_d]
        kv_blocks = [kv_coeffs_u, kv_coeffs_v, kv_coeffs_d]
        block_sizes = [block_dim, block_dim, block_dim]
    elif coordinate_mode == "uv_mixed_d_axial":
        if mixed_freqs is None or mixed_freqs.shape[-2:] != (
            2 * head_dim // 6,
            2,
        ):
            raise ValueError(
                "Unexpected hybrid Mixed frequency shape "
                f"{getattr(mixed_freqs, 'shape', None)}."
            )
        q_uv = _rope_precompute_mixed_coeffs_batch(
            q_u[:, None].expand(-1, mixed_freqs.shape[0], -1),
            q_v[:, None].expand(-1, mixed_freqs.shape[0], -1),
            mixed_freqs,
        )
        kv_uv = _prepare_gmm_mixed_factor_coeffs(
            kv_positions[..., :2],
            uvd_factors[..., :2],
            kv_weights,
            mixed_freqs,
        )
        q_d = _prepare_gmm_axial_factor_coeffs(
            q_positions[..., 2],
            q_factors[..., 2],
            q_weights,
            depth_axial_freqs,
        )
        kv_d = _prepare_gmm_axial_factor_coeffs(
            kv_positions[..., 2],
            uvd_factors[..., 2],
            kv_weights,
            depth_axial_freqs,
        )
        q_blocks = [q_uv, q_d]
        kv_blocks = [kv_uv, kv_d]
        block_sizes = [2 * block_dim, block_dim]
    else:
        if mixed_freqs is None or mixed_freqs.shape[-2:] != (head_dim // 2, 3):
            raise ValueError(
                "Unexpected 3D Mixed frequency shape "
                f"{getattr(mixed_freqs, 'shape', None)}."
            )
        q_full = _prepare_gmm_mixed_factor_coeffs(
            q_positions, q_factors, q_weights, mixed_freqs
        )
        kv_full = _prepare_gmm_mixed_factor_coeffs(
            kv_positions, uvd_factors, kv_weights, mixed_freqs
        )
        q_blocks = [q_full]
        kv_blocks = [kv_full]
        block_sizes = [head_dim]

    transforms_q = [
        (partial(_rope_apply_mixed_coeffs, coeffs=coeffs), size)
        for coeffs, size in zip(q_blocks, block_sizes)
    ]
    transforms_kv = [
        (partial(_rope_apply_mixed_coeffs, coeffs=coeffs), size)
        for coeffs, size in zip(kv_blocks, block_sizes)
    ]
    transforms_o = [
        (partial(_rope_apply_mixed_coeffs, coeffs=coeffs, inverse=True), size)
        for coeffs, size in zip(q_blocks, block_sizes)
    ]
    return (
        partial(_apply_block_diagonal, func_size_pairs=transforms_q),
        partial(_apply_block_diagonal, func_size_pairs=transforms_kv),
        partial(_apply_block_diagonal, func_size_pairs=transforms_o),
    )


############################################################################
# Project the finite source-patch footprint into query-view U/V covariance.
# Match a uniform patch with pixel std = patch_size / sqrt(12).
# Scale source tangents by mean depth and linearize only the final projection.
############################################################################
def _project_fixed_lateral_footprint_covariance(
    component_depths: torch.Tensor,
    projected_component_means: torch.Tensor,
    query_Ks: torch.Tensor,
    source_Ks_inv: torch.Tensor,
    camera_to_camera: torch.Tensor,
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a uniform-patch-matched Direct-3D Gaussian to query U/V.

    The source-image Gaussian has fixed per-axis standard deviations
    ``patch_width/sqrt(12)`` and ``patch_height/sqrt(12)`` in pixel units.
    Multiplying its two camera-space tangent vectors by the component mean
    depth makes the metric footprint grow linearly with depth.  The rigid
    source-to-query transform is exact; only the final pinhole projection is
    linearized at the component mean.  This is the Direct-3D construction

        Sigma = mu_D^2 sigma_U^2 t_U t_U^T
              + mu_D^2 sigma_V^2 t_V t_V^T,

    deliberately without a ``sigma_D^2 sigma_UV^2`` cross term.  The return
    values are the two marginal variances and their covariance in URoPE's
    query patch-grid coordinates, all in FP32.
    """
    batch, cameras = source_Ks_inv.shape[:2]
    patch_num = patches_x * patches_y
    batch_queries = batch * cameras
    components = component_depths.shape[1]
    tokens = cameras * patch_num
    if source_Ks_inv.shape != (batch, cameras, 3, 3):
        raise ValueError(f"Unexpected inverse-intrinsics shape {source_Ks_inv.shape}.")
    if camera_to_camera.shape != (batch, cameras, cameras, 4, 4):
        raise ValueError(f"Unexpected camera transform shape {camera_to_camera.shape}.")
    if query_Ks.shape != (batch_queries, 3, 3):
        raise ValueError(f"Unexpected query intrinsics shape {query_Ks.shape}.")
    if projected_component_means.shape != (
        batch_queries,
        components,
        tokens,
        3,
    ):
        raise ValueError(
            "Unexpected projected component means shape "
            f"{projected_component_means.shape}."
        )

    device_type = projected_component_means.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        depths = component_depths.float().expand(
            batch_queries, components, tokens
        )
        Ks_inv = source_Ks_inv.float()
        rotations = camera_to_camera[..., :3, :3].float()

        # K^{-1}e_u and K^{-1}e_v are camera-space changes per source pixel.
        source_u_tangent = Ks_inv[..., :, 0]
        source_v_tangent = Ks_inv[..., :, 1]
        query_u_tangent = torch.einsum(
            "bqsij,bsj->bqsi", rotations, source_u_tangent
        )
        query_v_tangent = torch.einsum(
            "bqsij,bsj->bqsi", rotations, source_v_tangent
        )

        ############################################################################
        # Broadcast a source tangent over query cameras, Gaussian components and tokens.
        # Keep the same flattened camera/token ordering used by projected component means.
        ############################################################################
        def expand_source_tangent(tangent: torch.Tensor) -> torch.Tensor:
            tangent = tangent[:, :, :, None, :].expand(
                batch, cameras, cameras, patch_num, 3
            )
            return tangent.reshape(batch_queries, 1, tokens, 3)

        query_u_tangent = expand_source_tangent(query_u_tangent)
        query_v_tangent = expand_source_tangent(query_v_tangent)
        # Direct-3D lateral scale: mu_D times a fixed angular/pixel tangent.
        query_u_tangent = query_u_tangent * depths[..., None]
        query_v_tangent = query_v_tangent * depths[..., None]

        query_Ks_f = query_Ks.float()
        projected_u_tangent = torch.einsum(
            "bdti,bji->bdtj", query_u_tangent, query_Ks_f
        )
        projected_v_tangent = torch.einsum(
            "bdti,bji->bdtj", query_v_tangent, query_Ks_f
        )
        projected_means_f = projected_component_means.float()

        x_unclamped, dx_du = _urope_image_coordinate_and_depth_derivative(
            projected_means_f,
            projected_depth_derivative=projected_u_tangent,
            coordinate_index=0,
            coordinate_scale=(image_width / patches_x),
        )
        _, dx_dv = _urope_image_coordinate_and_depth_derivative(
            projected_means_f,
            projected_depth_derivative=projected_v_tangent,
            coordinate_index=0,
            coordinate_scale=(image_width / patches_x),
        )
        y_unclamped, dy_du = _urope_image_coordinate_and_depth_derivative(
            projected_means_f,
            projected_depth_derivative=projected_u_tangent,
            coordinate_index=1,
            coordinate_scale=(image_height / patches_y),
        )
        _, dy_dv = _urope_image_coordinate_and_depth_derivative(
            projected_means_f,
            projected_depth_derivative=projected_v_tangent,
            coordinate_index=1,
            coordinate_scale=(image_height / patches_y),
        )

        # Match the derivative of the deterministic URoPE clamp.  A saturated
        # output coordinate has zero local variance in that coordinate.
        x_active = ((x_unclamped > -64) & (x_unclamped < 96)).float()
        y_active = ((y_unclamped > -64) & (y_unclamped < 96)).float()
        dx_du = dx_du * x_active
        dx_dv = dx_dv * x_active
        dy_du = dy_du * y_active
        dy_dv = dy_dv * y_active

        sigma_u_pixels = (image_width / patches_x) / math.sqrt(12.0)
        sigma_v_pixels = (image_height / patches_y) / math.sqrt(12.0)
        x_from_u = dx_du * sigma_u_pixels
        x_from_v = dx_dv * sigma_v_pixels
        y_from_u = dy_du * sigma_u_pixels
        y_from_v = dy_dv * sigma_v_pixels
        variance_x = x_from_u.square() + x_from_v.square()
        variance_y = y_from_u.square() + y_from_v.square()
        covariance_xy = x_from_u * y_from_u + x_from_v * y_from_v
    return variance_x, variance_y, covariance_xy


############################################################################
# Differentiate projected patch-grid U/V with respect to source depth.
# The projection uses abs(z) + epsilon; coordinate_scale converts pixels to patches.
# Return the coordinate and its optional depth derivative for the legacy UV path.
############################################################################
def _urope_image_coordinate_and_depth_derivative(
    projected_xyz: torch.Tensor,
    projected_depth_derivative: Optional[torch.Tensor],
    coordinate_index: int,
    coordinate_scale: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Evaluate URoPE image position and its derivative along key depth."""
    if coordinate_index not in (0, 1):
        raise ValueError(f"Expected image coordinate 0 or 1, got {coordinate_index}.")
    projected_z = projected_xyz[..., 2]
    denominator = torch.abs(projected_z) + 1e-5
    coordinate = (
        projected_xyz[..., coordinate_index]
        / denominator
        / coordinate_scale
    )
    if projected_depth_derivative is None:
        return coordinate, None
    denominator_derivative = (
        torch.sign(projected_z) * projected_depth_derivative[..., 2]
    )
    derivative = (
        projected_depth_derivative[..., coordinate_index] * denominator
        - projected_xyz[..., coordinate_index] * denominator_derivative
    ) / denominator.square()
    return coordinate, derivative / coordinate_scale
