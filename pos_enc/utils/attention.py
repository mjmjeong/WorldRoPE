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

from pos_enc.utils.rotary import _init_random_2d_mixed_frequencies, _init_random_2d_mixed_frequencies_range, _init_random_3d_mixed_frequencies_range, _logspace_frequencies
from pos_enc.urope.multi_head_anchor import _prepare_apply_fns
from pos_enc.worldrope.token_with_gaussian_splatting import _prepare_apply_fns_uvd


class MultiViewRoPEAttention(torch.nn.Module):
    """URoPE attention."""

    def __init__(
        self,
        head_num: int,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        min_depth: float = 2.0,
        max_depth: float = 20.0,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
        group_size: int = 4,
        leaveout_head: int = 0,
        learned_anchor_mixture: bool = False,
        learned_anchor_gaussian_mixture: bool = False,
        learned_anchor_means: bool = False,
        fixed_lateral_footprint_kv: bool = False,
        mixed_frequencies: bool = False,
        coordinate_mode: str = "uv",
        num_layers: int = 0,
        mixed_init_seed: int = 42,
        uv_frequency_min: float = 100.0 ** (-11.0 / 12.0),
        uv_frequency_max: float = 1.0,
        depth_frequency_min: float = 2.0 * 100.0 ** (-11.0 / 12.0),
        depth_frequency_max: float = 2.0,
        mixed_3d_frequency_min: float = 100.0 ** (-11.0 / 12.0),
        mixed_3d_frequency_max: float = 2.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height

        self.head_num = head_num
        self.leaveout_head = leaveout_head
        self.depth_num = (head_num - leaveout_head) // group_size
        self.group_size = group_size
        self.learned_anchor_mixture = learned_anchor_mixture
        self.learned_anchor_gaussian_mixture = learned_anchor_gaussian_mixture
        self.learned_anchor_means = learned_anchor_means
        self.fixed_lateral_footprint_kv = fixed_lateral_footprint_kv
        self.mixed_frequencies = mixed_frequencies
        self.coordinate_mode = coordinate_mode
        self.freq_base = freq_base
        if learned_anchor_gaussian_mixture and not learned_anchor_mixture:
            raise ValueError(
                "A learned anchor Gaussian mixture also requires mixture weights."
            )
        if learned_anchor_means and not learned_anchor_gaussian_mixture:
            raise ValueError("Learned anchor means require the Gaussian mixture.")

        self.max_depth = max_depth
        self.min_depth = min_depth

        valid_coordinate_modes = {
            "uv",
            "uvd_axial",
            "uv_mixed_d_axial",
            "uvd_mixed",
        }
        if coordinate_mode not in valid_coordinate_modes:
            raise ValueError(
                f"Unknown URoPE coordinate mode {coordinate_mode!r}; expected "
                f"one of {sorted(valid_coordinate_modes)}."
            )
        if coordinate_mode != "uv" and mixed_frequencies:
            raise ValueError(
                "Legacy 2D mixed_frequencies cannot be combined with a UVD mode."
            )
        if coordinate_mode != "uv":
            if head_dim % 6 != 0:
                raise ValueError(
                    f"UVD URoPE requires head_dim divisible by 6, got {head_dim}."
                )
            if not learned_anchor_gaussian_mixture:
                raise ValueError("UVD URoPE requires Gaussian anchor mixtures.")
            for label, lower, upper in (
                ("uv", uv_frequency_min, uv_frequency_max),
                ("depth", depth_frequency_min, depth_frequency_max),
                ("3d mixed", mixed_3d_frequency_min, mixed_3d_frequency_max),
            ):
                if not 0.0 < lower <= upper:
                    raise ValueError(
                        f"Invalid {label} frequency range [{lower}, {upper}]."
                    )

            axial_frequency_count = head_dim // 6
            self.register_buffer(
                "uv_axial_freqs",
                _logspace_frequencies(
                    uv_frequency_min, uv_frequency_max, axial_frequency_count
                ),
            )
            self.register_buffer(
                "depth_axial_freqs",
                _logspace_frequencies(
                    depth_frequency_min,
                    depth_frequency_max,
                    axial_frequency_count,
                ),
            )
        else:
            self.register_buffer("uv_axial_freqs", None)
            self.register_buffer("depth_axial_freqs", None)

        if self.mixed_frequencies:
            if leaveout_head != 0:
                raise ValueError("Mixed URoPE does not support leaveout heads.")
            if num_layers <= 0:
                raise ValueError("Mixed URoPE requires a positive num_layers.")
            self.mixed_freqs = torch.nn.Parameter(
                _init_random_2d_mixed_frequencies(
                    num_layers=num_layers,
                    num_heads=head_num,
                    head_dim=head_dim,
                    freq_base=freq_base,
                    seed=mixed_init_seed,
                )
            )
        elif coordinate_mode == "uv_mixed_d_axial":
            if leaveout_head != 0 or num_layers <= 0:
                raise ValueError(
                    "UV-Mixed + axial-D requires no leaveout heads and positive layers."
                )
            self.mixed_freqs = torch.nn.Parameter(
                _init_random_2d_mixed_frequencies_range(
                    num_layers=num_layers,
                    num_heads=head_num,
                    num_magnitudes=head_dim // 6,
                    frequency_min=uv_frequency_min,
                    frequency_max=uv_frequency_max,
                    seed=mixed_init_seed,
                )
            )
        elif coordinate_mode == "uvd_mixed":
            if leaveout_head != 0 or num_layers <= 0:
                raise ValueError(
                    "3D-Mixed UVD requires no leaveout heads and positive layers."
                )
            self.mixed_freqs = torch.nn.Parameter(
                _init_random_3d_mixed_frequencies_range(
                    num_layers=num_layers,
                    num_heads=head_num,
                    num_magnitudes=head_dim // 6,
                    frequency_min=mixed_3d_frequency_min,
                    frequency_max=mixed_3d_frequency_max,
                    seed=mixed_init_seed,
                )
            )
        else:
            self.register_parameter("mixed_freqs", None)

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
        Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
        mod_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        mixed_freqs = None
        if self.mixed_frequencies or self.coordinate_mode in (
            "uv_mixed_d_axial",
            "uvd_mixed",
        ):
            if mod_id is None or not 0 <= mod_id < self.mixed_freqs.shape[0]:
                raise ValueError(
                    f"Mixed URoPE requires a valid layer id, got {mod_id}."
                )
            mixed_freqs = self.mixed_freqs[mod_id]
        return multiview_rope_dot_product_attention(
            q,
            k,
            v,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            group_size=self.group_size,
            depth_num=self.depth_num,
            max_depth=self.max_depth,
            min_depth=self.min_depth,
            leaveout_head=self.leaveout_head,
            learned_anchor_mixture=self.learned_anchor_mixture,
            learned_anchor_gaussian_mixture=(
                self.learned_anchor_gaussian_mixture
            ),
            learned_anchor_means=self.learned_anchor_means,
            fixed_lateral_footprint_kv=self.fixed_lateral_footprint_kv,
            mixed_freqs=mixed_freqs,
            coordinate_mode=self.coordinate_mode,
            uv_axial_freqs=self.uv_axial_freqs,
            depth_axial_freqs=self.depth_axial_freqs,
            **kwargs,
        )


def multiview_rope_dot_product_attention(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
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
    leaveout_head: int = 0,
    learned_anchor_mixture: bool = False,
    learned_anchor_gaussian_mixture: bool = False,
    learned_anchor_means: bool = False,
    fixed_lateral_footprint_kv: bool = False,
    anchor_weights: Optional[torch.Tensor] = None,
    anchor_sigmas: Optional[torch.Tensor] = None,
    anchor_means: Optional[torch.Tensor] = None,
    mixed_freqs: Optional[torch.Tensor] = None,
    coordinate_mode: str = "uv",
    uv_axial_freqs: Optional[torch.Tensor] = None,
    depth_axial_freqs: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Like F.scaled_dot_product_attention, with URoPE-style ray / image-plane RoPE.

    Assumes self-attention shapes and ``seqlen == cameras * patches_x * patches_y``,
    with tokens ordered so ``seqlen`` can be viewed as ``(cameras, patches_x, patches_y)``.
    """
    # We're going to assume self-attention: all inputs are the same shape.
    (batch, num_heads, seqlen, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
    assert seqlen == cameras * patches_x * patches_y
    if learned_anchor_mixture:
        assert anchor_weights is not None
        assert anchor_weights.shape == (batch, seqlen, depth_num)
    else:
        assert anchor_weights is None
    if learned_anchor_gaussian_mixture:
        assert learned_anchor_mixture
        assert anchor_sigmas is not None
        assert anchor_sigmas.shape == (batch, seqlen, depth_num)
    else:
        assert anchor_sigmas is None
    if learned_anchor_means:
        assert learned_anchor_gaussian_mixture
        assert anchor_means is not None
        assert anchor_means.shape == (batch, seqlen, depth_num)
    else:
        assert anchor_means is None

    prepare_function = (
        _prepare_apply_fns
        if coordinate_mode == "uv"
        else _prepare_apply_fns_uvd
    )
    apply_fn_q, apply_fn_kv, apply_fn_o = prepare_function(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
        group_size=group_size,
        depth_num=depth_num,
        max_depth=max_depth,
        min_depth=min_depth,
        anchor_weights=anchor_weights,
        anchor_sigmas=anchor_sigmas,
        anchor_means=anchor_means,
        fixed_lateral_footprint_kv=fixed_lateral_footprint_kv,
        mixed_freqs=mixed_freqs,
        **(
            {}
            if coordinate_mode == "uv"
            else {
                "coordinate_mode": coordinate_mode,
                "uv_axial_freqs": uv_axial_freqs,
                "depth_axial_freqs": depth_axial_freqs,
            }
        ),
    )

    patch_num = patches_x * patches_y
    q = q.reshape(batch, num_heads, cameras, patch_num, head_dim)
    q = q.permute(0, 2, 1, 3, 4).reshape(batch*cameras, num_heads, patch_num, head_dim)  # [batch*cameras, num_heads, patch_num, head_dim]
    k = k.reshape(batch, num_heads, seqlen, head_dim)
    k = k[:, None].repeat(1, cameras, 1, 1, 1)
    k = k.flatten(0, 1)  # [batch*cameras, num_heads, seqlen, head_dim]
    v = v.reshape(batch, num_heads, seqlen, head_dim)
    v = v[:, None].repeat(1, cameras, 1, 1, 1)
    v = v.flatten(0, 1)  # [batch*cameras, num_heads, seqlen, head_dim]

    if leaveout_head > 0:
        q_1 = q[:, :-leaveout_head]
        q_2 = q[:, -leaveout_head:]
        k_1 = k[:, :-leaveout_head]
        k_2 = k[:, -leaveout_head:]
        v_1 = v[:, :-leaveout_head]
        v_2 = v[:, -leaveout_head:]

        q_1 = apply_fn_q(q_1)
        k_1 = apply_fn_kv(k_1)
        v_1 = apply_fn_kv(v_1)

        q = torch.cat([q_1, q_2], dim=1)
        k = torch.cat([k_1, k_2], dim=1)
        v = torch.cat([v_1, v_2], dim=1)
    else:
        q = apply_fn_q(q)
        k = apply_fn_kv(k)
        v = apply_fn_kv(v)

    out = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        **kwargs,
    )
    if leaveout_head > 0:
        out_1 = out[:, :-leaveout_head]
        out_2 = out[:, -leaveout_head:]
        out_1 = apply_fn_o(out_1)
        out = torch.cat([out_1, out_2], dim=1)
    else:
        out = apply_fn_o(out)  # [batch*cameras, num_heads, patch_num, head_dim]
    out = out.reshape(batch, cameras, num_heads, patch_num, head_dim)
    out = out.permute(0, 2, 1, 3, 4).reshape(batch, num_heads, seqlen, head_dim)
    assert out.shape == (batch, num_heads, seqlen, head_dim)

    return out
