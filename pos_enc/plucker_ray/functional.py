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

import time
from collections import namedtuple
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

Camera = namedtuple("Camera", ["K", "camtoworld", "width", "height"])



############################################################################
# Generate per-pixel world-space ray origins and directions from intrinsics
# and camera-to-world poses.
############################################################################
def camera_to_raymap(
    Ks: Tensor,
    camtoworlds: Tensor,
    height: int,
    width: int,
    downscale: int = 1,
    include_ups: bool = False,
):
    """Construct the raymap from the camera intrinsics and extrinsics.

    Note: This function expects OpenCV camera coordinates.

    Args:
        Ks: The camera intrinsics tensor with shape (..., 3, 3).
        camtoworlds: The camera extrinsics tensor with shape (..., 4, 4).
        height: The height of original image corresponding to intrinsics.
        width: The width of original image corresponding to intrinsics.
        downscale: Downscale factor for the raymap.
        include_ups: Whether to include the up direction in the raymap.

    Returns:
        The raymap tensor with shape (..., H, W, 6).
    """
    assert Ks.shape[-2:] == (3, 3), "Expected Ks to have shape (..., 3, 3)."
    assert camtoworlds.shape[-2:] == (
        4,
        4,
    ), "Expected camtoworlds to have shape (..., 4, 4)."
    assert width % downscale == 0, "Expected width to be divisible by downscale."
    assert height % downscale == 0, "Expected height to be divisible by downscale."

    # Downscale the intrinsics.
    Ks = torch.stack(
        [
            Ks[..., 0, :] / downscale,
            Ks[..., 1, :] / downscale,
            Ks[..., 2, :],
        ],
        dim=-2,
    )  # [..., 3, 3]
    width //= downscale
    height //= downscale

    # Construct pixel coordinates
    x, y = torch.meshgrid(
        torch.arange(width, device=Ks.device),
        torch.arange(height, device=Ks.device),
        indexing="xy",
    )  # [H, W]
    coords = torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1)  # [H, W, 3]

    # To camera coordinates [..., H, W, 3]
    dirs = torch.einsum("...ij,...hwj->...hwi", Ks.inverse(), coords)

    # To world coordinates [..., H, W, 3]
    dirs = torch.einsum("...ij,...hwj->...hwi", camtoworlds[..., :3, :3], dirs)
    dirs = F.normalize(dirs, p=2, dim=-1)

    # Camera origin in world coordinates [..., H, W, 3]
    origins = torch.broadcast_to(camtoworlds[..., None, None, :3, -1], dirs.shape)

    if include_ups:
        # Extract the up direction (second column)
        ups = torch.broadcast_to(camtoworlds[..., None, None, :3, 1], dirs.shape)
        ups = F.normalize(ups, p=2, dim=-1)
        return torch.cat([origins, dirs, ups], dim=-1)
    else:
        return torch.cat([origins, dirs], dim=-1)  # [..., H, W, 6]


############################################################################
# Convert ray origins/directions to Plucker direction and moment
# coordinates for input conditioning.
############################################################################
def raymap_to_plucker(raymap: Tensor) -> Tensor:
    """Convert raymap to Plücker coordinates.

    Args:
        raymap: The raymap tensor with shape (..., H, W, 6).

    Returns:
        The Plücker coordinates tensor with shape (..., H, W, 3).
    """
    assert raymap.shape[-1] == 6, "Expected raymap to have shape (..., H, W, 6)."
    ray_origins, ray_directions = torch.split(raymap, [3, 3], dim=-1)
    # Normalize ray directions to unit vectors
    ray_directions = F.normalize(ray_directions, p=2, dim=-1)
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    return torch.cat([ray_directions, plucker_normal], dim=-1)
