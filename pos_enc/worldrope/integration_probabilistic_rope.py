"""Closed-form probabilistic RoPE integration using Gaussian characteristic functions.

For a Gaussian component, E[exp(i f^T X)] = exp(i f^T mu - 0.5 f^T Sigma f).
Mixture weights combine these expected operators after component integration.
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



############################################################################
# Integrate one axial rotary frequency over a Gaussian mixture in FP32.
# Sum pi_g * exp(-0.5 * phase_variance_g) * (cos, sin)(mean_phase_g).
# Covariance factors avoid explicit covariance matrices and unstable cross terms.
############################################################################
def _gmm_axial_factor_coeffs_fp32(
    positions: torch.Tensor,
    factors: torch.Tensor,
    weights: torch.Tensor,
    frequencies: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Streaming scalar-GMM characteristic sum from std-vector factors."""
    frequency = frequencies[None, None, None, :]
    cosine = torch.zeros(
        (positions.shape[0], 1, positions.shape[2], frequencies.shape[0]),
        dtype=positions.dtype,
        device=positions.device,
    )
    sine = torch.zeros_like(cosine)
    for component_index in range(positions.shape[1]):
        position = positions[:, component_index, :, None][:, None]
        component_factors = factors[:, component_index]
        weight = weights[:, component_index, :, None][:, None]
        angle = position * frequency
        projected_factors = (
            component_factors[:, None, :, :, None] * frequency[:, :, :, None]
        )
        # Once an individual phase std exceeds sqrt(80), its Gaussian
        # attenuation is already below useful FP32 precision.  Saturating the
        # std before squaring prevents 0*inf derivatives at pinhole
        # singularities without changing any numerically relevant value.
        projected_factors = projected_factors.clamp(-math.sqrt(80.0), math.sqrt(80.0))
        phase_variance = projected_factors.square().sum(dim=-2)
        amplitude = weight * torch.exp(-0.5 * phase_variance)
        cosine = cosine + amplitude * torch.cos(angle)
        sine = sine + amplitude * torch.sin(angle)
    return cosine, sine


############################################################################
# Validate scalar GMM shapes [B,G,T] and covariance factors [B,G,T,L].
# Disable autocast before evaluating the axial expected rotary coefficients.
############################################################################
def _prepare_gmm_axial_factor_coeffs(
    positions: torch.Tensor,
    factors: torch.Tensor,
    weights: torch.Tensor,
    frequencies: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if positions.shape != weights.shape:
        raise ValueError("Scalar GMM positions/weights must match [B,D,S].")
    if factors.ndim != 4 or factors.shape[:3] != positions.shape:
        raise ValueError("Scalar GMM factors must have shape [B,D,S,L].")
    device_type = positions.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return _gmm_axial_factor_coeffs_fp32(
            positions.float(),
            factors.float(),
            weights.float(),
            frequencies.float(),
        )


############################################################################
# Integrate mixed-direction frequencies over vector-valued Gaussian supports.
# For each frequency f, use f dot mean and sum_l (f dot factor_l)^2.
# Stream component sums to avoid materializing all component/frequency tensors.
############################################################################
def _gmm_mixed_factor_coeffs_fp32(
    positions: torch.Tensor,
    factors: torch.Tensor,
    weights: torch.Tensor,
    frequencies: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Streaming Gaussian characteristic sum from covariance factors.

    ``factors[..., l, :]`` are independent standard-deviation vectors, so
    ``Sigma = sum_l factor_l factor_l^T``.  Evaluating
    ``f^T Sigma f = sum_l (f^T factor_l)^2`` directly is both exact and more
    stable than materializing covariance entries and adding signed cross
    terms, which can produce ``inf - inf`` near a pinhole singularity.
    """
    frequency = frequencies[None, :, None]
    cosine = torch.zeros(
        (
            positions.shape[0],
            frequencies.shape[0],
            positions.shape[2],
            frequencies.shape[1],
        ),
        dtype=positions.dtype,
        device=positions.device,
    )
    sine = torch.zeros_like(cosine)
    for component_index in range(positions.shape[1]):
        mean = positions[:, component_index, :, None, :][:, None]
        component_factors = factors[:, component_index]
        weight = weights[:, component_index, :, None][:, None]
        angle = (mean * frequency).sum(dim=-1)
        projected_factors = torch.einsum(
            "bslc,hfc->bhsfl",
            component_factors,
            frequencies,
        )
        projected_factors = projected_factors.clamp(
            -math.sqrt(80.0), math.sqrt(80.0)
        )
        phase_variance = projected_factors.square().sum(dim=-1)
        amplitude = weight * torch.exp(-0.5 * phase_variance)
        cosine = cosine + amplitude * torch.cos(angle)
        sine = sine + amplitude * torch.sin(angle)
    return cosine, sine


############################################################################
# Validate vector GMM/frequency shapes and evaluate mixed coefficients in FP32.
# This path retains the off-diagonal covariance contribution through factors.
############################################################################
def _prepare_gmm_mixed_factor_coeffs(
    positions: torch.Tensor,
    factors: torch.Tensor,
    weights: torch.Tensor,
    frequencies: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if positions.ndim != 4:
        raise ValueError(f"Expected GMM positions [B,D,S,C], got {positions.shape}.")
    coordinates = positions.shape[-1]
    if factors.ndim != 5 or factors.shape[:3] != positions.shape[:3]:
        raise ValueError(f"Unexpected Gaussian-factor shape {factors.shape}.")
    if factors.shape[-1] != coordinates:
        raise ValueError(
            f"Factor coordinate count {factors.shape[-1]} does not match "
            f"positions {coordinates}."
        )
    if weights.shape != positions.shape[:-1]:
        raise ValueError(f"Unexpected mixture-weight shape {weights.shape}.")
    if frequencies.ndim != 3 or frequencies.shape[-1] != coordinates:
        raise ValueError(f"Unexpected frequency shape {frequencies.shape}.")
    device_type = positions.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return _gmm_mixed_factor_coeffs_fp32(
            positions.float(),
            factors.float(),
            weights.float(),
            frequencies.float(),
        )


############################################################################
# Evaluate expected 2D Mixed-RoPE from projected means and full U/V covariance.
# The covariance_xy term couples the two image axes in the phase variance.
############################################################################
def _rope_precompute_mixed_gaussian_coeffs_batch(
    positions_x: torch.Tensor,
    positions_y: torch.Tensor,
    variance_x: torch.Tensor,
    variance_y: torch.Tensor,
    covariance_xy: torch.Tensor,
    frequencies: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expected per-head Mixed-RoPE for a projected 2D Gaussian."""
    if not (
        positions_x.shape
        == positions_y.shape
        == variance_x.shape
        == variance_y.shape
        == covariance_xy.shape
    ) or positions_x.ndim != 3:
        raise ValueError("Mixed Gaussian inputs must match [B,H,T].")
    expected_shape = (positions_x.shape[1], frequencies.shape[1], 2)
    if frequencies.shape != expected_shape:
        raise ValueError(
            f"Unexpected mixed frequency shape {frequencies.shape}; "
            f"expected {expected_shape}."
        )
    device_type = positions_x.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        x = positions_x.float()[..., None]
        y = positions_y.float()[..., None]
        var_x = variance_x.float()[..., None]
        var_y = variance_y.float()[..., None]
        cov_xy = covariance_xy.float()[..., None]
        freq_x = frequencies.float()[None, :, None, :, 0]
        freq_y = frequencies.float()[None, :, None, :, 1]
        angles = x * freq_x + y * freq_y
        phase_variance = (
            freq_x.square() * var_x
            + 2.0 * freq_x * freq_y * cov_xy
            + freq_y.square() * var_y
        ).clamp_min(0.0)
        attenuation = torch.exp(-0.5 * phase_variance)
        return attenuation * torch.cos(angles), attenuation * torch.sin(angles)


############################################################################
# Evaluate 2D Gaussian expected rotations separately for each depth component.
# Preserve the component axis so mixture weights can be applied afterwards.
############################################################################
def _rope_precompute_mixed_gaussian_anchor_coeffs_batch(
    positions_x: torch.Tensor,
    positions_y: torch.Tensor,
    variance_x: torch.Tensor,
    variance_y: torch.Tensor,
    covariance_xy: torch.Tensor,
    frequencies: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expected Mixed-RoPE for every [B,D,T] Gaussian component."""
    if not (
        positions_x.shape
        == positions_y.shape
        == variance_x.shape
        == variance_y.shape
        == covariance_xy.shape
    ) or positions_x.ndim != 3:
        raise ValueError("Mixed Gaussian anchor inputs must match [B,D,T].")
    if frequencies.ndim != 3 or frequencies.shape[-1] != 2:
        raise ValueError("Mixed frequencies must have shape [H,F,2].")
    device_type = positions_x.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        x = positions_x.float()[:, None, :, :, None]
        y = positions_y.float()[:, None, :, :, None]
        var_x = variance_x.float()[:, None, :, :, None]
        var_y = variance_y.float()[:, None, :, :, None]
        cov_xy = covariance_xy.float()[:, None, :, :, None]
        freqs = frequencies.float()[None, :, None, None, :, :]
        freq_x = freqs[..., 0]
        freq_y = freqs[..., 1]
        angles = x * freq_x + y * freq_y
        phase_variance = (
            freq_x.square() * var_x
            + 2.0 * freq_x * freq_y * cov_xy
            + freq_y.square() * var_y
        ).clamp_min(0.0)
        attenuation = torch.exp(-0.5 * phase_variance)
        return attenuation * torch.cos(angles), attenuation * torch.sin(angles)


############################################################################
# Prepare weights and FP32 inputs for fused projected-Gaussian integration.
# Accumulate the mixture without storing the large per-component rotary tensor.
############################################################################
def _rope_precompute_mixed_gaussian_anchor_mixture_coeffs_batch(
    positions_x: torch.Tensor,
    positions_y: torch.Tensor,
    variance_x: torch.Tensor,
    variance_y: torch.Tensor,
    covariance_xy: torch.Tensor,
    frequencies: torch.Tensor,
    anchor_weights: torch.Tensor,
    num_cameras: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate and mix full-covariance Gaussian anchors without a D tensor.

    This is mathematically identical to calling
    ``_rope_precompute_mixed_gaussian_anchor_coeffs_batch`` and then
    ``_mix_anchor_mixed_rope_coeffs``.  Accumulating the four components in a
    compiled FP32 kernel avoids materializing the large ``[B*Q,H,D,T,F]``
    cosine, sine, phase, and attenuation tensors.
    """
    if not (
        positions_x.shape
        == positions_y.shape
        == variance_x.shape
        == variance_y.shape
        == covariance_xy.shape
    ) or positions_x.ndim != 3:
        raise ValueError("Mixed Gaussian anchor inputs must match [B,D,T].")
    if frequencies.ndim != 3 or frequencies.shape[-1] != 2:
        raise ValueError("Mixed frequencies must have shape [H,F,2].")

    expanded_weights = _expand_anchor_token_values(
        anchor_weights,
        num_cameras=num_cameras,
    )
    if expanded_weights.shape != positions_x.shape:
        raise ValueError(
            "Expanded anchor weights and Gaussian components do not align: "
            f"{expanded_weights.shape} vs {positions_x.shape}."
        )

    device_type = positions_x.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return _rope_precompute_mixed_gaussian_anchor_mixture_coeffs_fp32(
            positions_x.float(),
            positions_y.float(),
            variance_x.float(),
            variance_y.float(),
            covariance_xy.float(),
            frequencies.float(),
            expanded_weights.float(),
        )


############################################################################
# Compiled FP32 characteristic-function sum for four projected 2D Gaussians.
# Apply covariance attenuation before combining the component rotations.
############################################################################
@torch.compile(fullgraph=True, dynamic=False)
def _rope_precompute_mixed_gaussian_anchor_mixture_coeffs_fp32(
    positions_x: torch.Tensor,
    positions_y: torch.Tensor,
    variance_x: torch.Tensor,
    variance_y: torch.Tensor,
    covariance_xy: torch.Tensor,
    frequencies: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compiled characteristic-function sum for four projected 2D Gaussians."""
    frequency_x = frequencies[None, :, None, :, 0]
    frequency_y = frequencies[None, :, None, :, 1]
    cosine = torch.zeros(
        (
            positions_x.shape[0],
            frequencies.shape[0],
            positions_x.shape[2],
            frequencies.shape[1],
        ),
        dtype=positions_x.dtype,
        device=positions_x.device,
    )
    sine = torch.zeros_like(cosine)
    for component_index in range(positions_x.shape[1]):
        x = positions_x[:, component_index, :, None][:, None]
        y = positions_y[:, component_index, :, None][:, None]
        var_x = variance_x[:, component_index, :, None][:, None]
        var_y = variance_y[:, component_index, :, None][:, None]
        cov_xy = covariance_xy[:, component_index, :, None][:, None]
        weight = weights[:, component_index, :, None][:, None]

        angle = x * frequency_x + y * frequency_y
        phase_variance = (
            frequency_x.square() * var_x
            + 2.0 * frequency_x * frequency_y * cov_xy
            + frequency_y.square() * var_y
        ).clamp_min(0.0)
        amplitude = weight * torch.exp(-0.5 * phase_variance)
        cosine = cosine + amplitude * torch.cos(angle)
        sine = sine + amplitude * torch.sin(angle)
    return cosine, sine


############################################################################
# Combine component Mixed-RoPE coefficients with source-token mixture weights.
# Reduce [B*Q,H,G,T,F] to [B*Q,H,T,F] in the shared camera/token ordering.
############################################################################
def _mix_anchor_mixed_rope_coeffs(
    coeffs_per_anchor: Tuple[torch.Tensor, torch.Tensor],
    anchor_weights: torch.Tensor,
    num_cameras: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weight [B*Q,H,D,T,F] component coefficients into [B*Q,H,T,F]."""
    cosine, sine = coeffs_per_anchor
    if cosine.shape != sine.shape or cosine.ndim != 5:
        raise ValueError("Mixed anchor coefficients must match [B*Q,H,D,T,F].")
    weights = _expand_anchor_token_values(
        anchor_weights,
        num_cameras=num_cameras,
    )[:, None, :, :, None]
    if cosine.shape[0] != weights.shape[0] or cosine.shape[2:4] != weights.shape[2:4]:
        raise ValueError("Mixed component coefficients and weights do not align.")
    weights = weights.to(dtype=cosine.dtype)
    return (
        (cosine * weights).sum(dim=2),
        (sine * weights).sum(dim=2),
    )


############################################################################
# Form the expected axial rotary operator from component rotations.
# Broadcast each source-token simplex to all query cameras before summation.
############################################################################
def _mix_anchor_rope_coeffs(
    coeffs_per_anchor: Tuple[torch.Tensor, torch.Tensor],
    anchor_weights: torch.Tensor,
    num_cameras: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mix projected anchor rotations as an expected RoPE operator.

    ``coeffs_per_anchor`` has one coefficient set per fixed URoPE depth
    anchor. ``anchor_weights`` has one simplex-valued distribution per source
    token.  Every query camera sees the same source-token distribution, while
    its projected anchor coefficients remain query-camera dependent.
    """
    cosine, sine = coeffs_per_anchor
    batch = anchor_weights.shape[0]
    seqlen = anchor_weights.shape[1]
    depth_num = anchor_weights.shape[2]
    assert cosine.shape == sine.shape
    assert cosine.shape[:3] == (batch * num_cameras, depth_num, seqlen)

    # [B,T,D] -> [B*Q,D,T,1], where Q is the query-camera axis.
    weights = _expand_anchor_token_values(
        anchor_weights,
        num_cameras=num_cameras,
    ).unsqueeze(-1)
    weights = weights.to(dtype=cosine.dtype)

    # The singleton head axis broadcasts the same expected rotation operator
    # to every attention head.
    mixed_cosine = (cosine * weights).sum(dim=1, keepdim=True)
    mixed_sine = (sine * weights).sum(dim=1, keepdim=True)
    return mixed_cosine, mixed_sine


############################################################################
# Broadcast token-wise component values to every query-camera frame.
# Convert [B,T,G] into [B*Q,G,T] without changing token/component semantics.
############################################################################
def _expand_anchor_token_values(
    values: torch.Tensor,
    num_cameras: int,
) -> torch.Tensor:
    """Broadcast source-token component values over all query cameras.

    Input ordering is ``[batch, source_token, component]``.  URoPE evaluates
    every source K/V token from every query camera, so the result is ordered as
    ``[batch * query_camera, component, source_token]``.
    """
    batch, seqlen, depth_num = values.shape
    expanded = values[:, None].expand(
        batch, num_cameras, seqlen, depth_num
    )
    expanded = expanded.reshape(batch * num_cameras, seqlen, depth_num)
    return expanded.permute(0, 2, 1)


############################################################################
# Integrate axial RoPE over scalar Gaussian means and variances.
# Multiply deterministic sine/cosine terms by covariance-dependent attenuation.
############################################################################
def _rope_precompute_gaussian_coeffs_batch(
    positions: torch.Tensor,
    variances: torch.Tensor,
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expected axial RoPE coefficients for first-order Gaussians."""
    assert positions.shape == variances.shape
    assert len(positions.shape) == 3
    assert feat_dim % 2 == 0
    num_heads = positions.shape[1]
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[
                None, None, None, :
            ]
            / num_freqs
        )
    )
    freqs = freqs.to(dtype=positions.dtype)
    angles = positions[..., None] * freqs
    attenuation = torch.exp(
        -0.5 * variances[..., None] * freqs.square()
    )
    assert angles.shape == attenuation.shape == (
        positions.shape[0],
        num_heads,
        positions.shape[2],
        num_freqs,
    )
    return attenuation * torch.cos(angles), attenuation * torch.sin(angles)
