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



def _prepare_explicit_axial_coeffs(
    positions: torch.Tensor,
    frequencies: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic axial coefficients for [B,H,S] positions."""
    if positions.ndim != 3 or frequencies.ndim != 1:
        raise ValueError("Axial positions/frequencies must be [B,H,S] and [F].")
    device_type = positions.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        angles = positions.float()[..., None] * frequencies.float()[None, None, None]
        return torch.cos(angles), torch.sin(angles)


def _init_random_2d_mixed_frequencies(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    freq_base: float,
    seed: int,
) -> torch.Tensor:
    """Initialize per-layer/head 2D Mixed-RoPE with theta-based magnitudes."""
    if head_dim % 4 != 0:
        raise ValueError(f"Mixed URoPE requires head_dim % 4 == 0, got {head_dim}.")
    if freq_base <= 0:
        raise ValueError(f"freq_base must be positive, got {freq_base}.")

    num_base_freqs = head_dim // 4
    magnitudes = freq_base ** (
        -torch.arange(num_base_freqs, dtype=torch.float32) / num_base_freqs
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        angles = torch.rand(num_layers, num_heads, 1) * (2.0 * math.pi)

    first_axis = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    second_axis = torch.stack([-torch.sin(angles), torch.cos(angles)], dim=-1)
    first_vectors = first_axis * magnitudes.reshape(1, 1, -1, 1)
    second_vectors = second_axis * magnitudes.reshape(1, 1, -1, 1)
    frequencies = torch.cat([first_vectors, second_vectors], dim=-2)
    expected_shape = (num_layers, num_heads, head_dim // 2, 2)
    if frequencies.shape != expected_shape:
        raise AssertionError((frequencies.shape, expected_shape))
    return frequencies.contiguous()


def _logspace_frequencies(
    frequency_min: float,
    frequency_max: float,
    count: int,
) -> torch.Tensor:
    """Return descending log-uniform frequencies with exact endpoints."""
    if count <= 0:
        raise ValueError(f"Frequency count must be positive, got {count}.")
    if not 0.0 < frequency_min <= frequency_max:
        raise ValueError(
            f"Expected 0 < min <= max, got {frequency_min}, {frequency_max}."
        )
    if count == 1:
        return torch.tensor([frequency_max], dtype=torch.float32)
    return torch.exp(
        torch.linspace(
            math.log(frequency_max),
            math.log(frequency_min),
            count,
            dtype=torch.float32,
        )
    )


def _init_random_2d_mixed_frequencies_range(
    num_layers: int,
    num_heads: int,
    num_magnitudes: int,
    frequency_min: float,
    frequency_max: float,
    seed: int,
) -> torch.Tensor:
    """Initialize two orthogonal SO(2) axes times one shared spectrum."""
    magnitudes = _logspace_frequencies(
        frequency_min, frequency_max, num_magnitudes
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        angles = torch.rand(num_layers, num_heads, 1) * (2.0 * math.pi)
    first_axis = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    second_axis = torch.stack([-torch.sin(angles), torch.cos(angles)], dim=-1)
    frequencies = torch.cat(
        [
            first_axis * magnitudes.reshape(1, 1, -1, 1),
            second_axis * magnitudes.reshape(1, 1, -1, 1),
        ],
        dim=-2,
    )
    expected_shape = (num_layers, num_heads, 2 * num_magnitudes, 2)
    if frequencies.shape != expected_shape:
        raise AssertionError((frequencies.shape, expected_shape))
    return frequencies.contiguous()


def _init_random_3d_mixed_frequencies_range(
    num_layers: int,
    num_heads: int,
    num_magnitudes: int,
    frequency_min: float,
    frequency_max: float,
    seed: int,
) -> torch.Tensor:
    """Initialize one proper SO(3) frame per layer/head.

    Each of the three orthogonal axes receives the same log-uniform magnitude
    spectrum.  The returned vectors are unconstrained learnable parameters;
    SO(3) is an initialization property only.
    """
    magnitudes = _logspace_frequencies(
        frequency_min, frequency_max, num_magnitudes
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        random_matrices = torch.randn(num_layers, num_heads, 3, 3)
        frames, upper = torch.linalg.qr(random_matrices)
        diagonal_sign = torch.sign(
            torch.diagonal(upper, dim1=-2, dim2=-1)
        )
        diagonal_sign = torch.where(
            diagonal_sign == 0,
            torch.ones_like(diagonal_sign),
            diagonal_sign,
        )
        frames = frames * diagonal_sign.unsqueeze(-2)
        determinant_sign = torch.where(
            torch.linalg.det(frames) < 0,
            -torch.ones((), dtype=frames.dtype),
            torch.ones((), dtype=frames.dtype),
        )
        frames[..., :, -1] = (
            frames[..., :, -1] * determinant_sign.unsqueeze(-1)
        )
    axes = frames.transpose(-1, -2)
    frequencies = (
        axes.unsqueeze(-2)
        * magnitudes.reshape(1, 1, 1, num_magnitudes, 1)
    ).reshape(num_layers, num_heads, 3 * num_magnitudes, 3)
    return frequencies.contiguous()


def _rope_precompute_mixed_coeffs_batch(
    positions_x: torch.Tensor,
    positions_y: torch.Tensor,
    frequencies: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-head 2D Mixed-RoPE coefficients in FP32."""
    if positions_x.shape != positions_y.shape or positions_x.ndim != 3:
        raise ValueError(
            f"Expected matching [B,H,S] positions, got "
            f"{positions_x.shape} and {positions_y.shape}."
        )
    expected_shape = (positions_x.shape[1], frequencies.shape[1], 2)
    if frequencies.shape != expected_shape:
        raise ValueError(
            f"Unexpected mixed frequency shape {frequencies.shape}; "
            f"expected {expected_shape}."
        )

    device_type = positions_x.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        x = positions_x.float()
        y = positions_y.float()
        freqs = frequencies.float()
        angles = (
            x[..., None] * freqs[None, :, None, :, 0]
            + y[..., None] * freqs[None, :, None, :, 1]
        )
        return torch.cos(angles), torch.sin(angles)


def _rope_precompute_mixed_anchor_coeffs_batch(
    positions_x: torch.Tensor,
    positions_y: torch.Tensor,
    derivative_x: Optional[torch.Tensor],
    derivative_y: Optional[torch.Tensor],
    sigmas: Optional[torch.Tensor],
    frequencies: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expected Mixed-RoPE coefficients for each depth-mixture component.

    Each component is a scalar Gaussian along the source ray. Under the same
    first-order projection used by Run 327, its Mixed phase derivative is
    ``theta_x * du/dD + theta_y * dv/dD``.
    """
    if positions_x.shape != positions_y.shape or positions_x.ndim != 3:
        raise ValueError("Mixed anchor positions must be matching [B,D,T] tensors.")
    if frequencies.ndim != 3 or frequencies.shape[-1] != 2:
        raise ValueError("Mixed frequencies must have shape [H,F,2].")
    gaussian = sigmas is not None
    if gaussian != (derivative_x is not None and derivative_y is not None):
        raise ValueError("Gaussian Mixed coefficients require both derivatives and sigma.")
    if gaussian and not (
        derivative_x.shape == derivative_y.shape == sigmas.shape == positions_x.shape
    ):
        raise ValueError("Projected derivatives and sigmas must match [B,D,T].")

    device_type = positions_x.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        x = positions_x.float()[:, None, :, :, None]
        y = positions_y.float()[:, None, :, :, None]
        freqs = frequencies.float()[None, :, None, None, :, :]
        angles = x * freqs[..., 0] + y * freqs[..., 1]
        attenuation = 1.0
        if gaussian:
            dx = derivative_x.float()[:, None, :, :, None]
            dy = derivative_y.float()[:, None, :, :, None]
            sigma = sigmas.float()[:, None, :, :, None]
            phase_derivative = dx * freqs[..., 0] + dy * freqs[..., 1]
            attenuation = torch.exp(-0.5 * sigma.square() * phase_derivative.square())
        return attenuation * torch.cos(angles), attenuation * torch.sin(angles)


def _rope_precompute_coeffs_batch(
    positions: torch.Tensor,  # (batch, num_heads, seqlen)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE coefficients."""
    assert len(positions.shape) == 3
    assert feat_dim % 2 == 0
    num_heads = positions.shape[1]
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    angles = positions[:, :, :, None] * freqs
    # Shape should be: `(batch, num_heads, seqlen, num_freqs)`; we're
    # broadcasting across `batch` and `num_heads`.
    assert angles.shape == (positions.shape[0], num_heads, positions.shape[2], num_freqs)
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""
    cos, sin = coeffs
    # We allow (cos, sin) to be either with shape (1, 1, seqlen, feat_dim // 2),
    # or (1, 1, seqlen_per_image, feat_dim // 2) and we repeat it to
    # match the shape of feats.
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


def _rope_apply_mixed_coeffs(
    feats: torch.Tensor,
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply FP32-computed Mixed-RoPE coefficients in the feature dtype."""
    typed_coeffs = tuple(coeff.to(dtype=feats.dtype) for coeff in coeffs)
    return _rope_apply_coeffs(feats, typed_coeffs, inverse=inverse)


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )

    assert out.shape == feats.shape, "Input/output shapes should match."
    return out
