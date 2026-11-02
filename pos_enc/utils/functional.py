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
from pos_enc.plucker_ray.functional import camera_to_raymap, raymap_to_plucker

Camera = namedtuple("Camera", ["K", "camtoworld", "width", "height"])


def random_SO3(batch_size: Tuple[int], device="cpu"):
    # Step 1: Generate a batch of random matrices of shape (batch_size, 3, 3)
    random_matrices = torch.randn((*batch_size, 3, 3), device=device)
    random_matrices = random_matrices.reshape(-1, 3, 3)

    # Step 2: Apply QR decomposition to each matrix in the batch
    # The `torch.linalg.qr` function works for batches of matrices in newer PyTorch versions
    q, r = torch.linalg.qr(random_matrices)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[..., None, :]

    # Step 3: Adjust for positive determinant in each matrix
    # Compute the determinants and find indices where the determinant is negative
    det_q = torch.det(q)
    negative_det_indices = det_q < 0

    # Flip the sign of the last column where determinant is negative
    q[negative_det_indices, :, 2] *= -1
    q = q.reshape(*batch_size, 3, 3)

    return q


def random_SE3(batch_size: Tuple[int], device="cpu"):
    random_matrices = torch.eye(4, device=device).repeat(*batch_size, 1, 1)
    random_matrices[..., :3, :3] = random_SO3(batch_size, device)
    random_matrices[..., :3, 3] = torch.randn(*batch_size, 3, device=device)
    return random_matrices


def patchify(x: Tensor, patch_size: int) -> Tensor:
    """Split an image tensor into patches.

    Args:
        x: The input image tensor with shape (..., H * P, W * P, C).
        patch_size: The size of the patch.

    Returns:
        The output tensor with shape (..., H * W, P * P * C).
    """
    assert (
        x.shape[-3] % patch_size == 0
    ), "Expected height to be divisible by patch_size."
    assert (
        x.shape[-2] % patch_size == 0
    ), "Expected width to be divisible by patch_size."

    x = rearrange(
        x, "... (h ph) (w pw) c -> ... (h w) (ph pw c)", ph=patch_size, pw=patch_size
    )
    return x


def unpatchify(x: Tensor, height: int, width: int, patch_size: int) -> Tensor:
    """Combine patches into an image tensor.

    Args:
        x: The input tensor with shape (..., H * W, P * P * C).
        height: The height of the original image.
        width: The width of the original image.
        patch_size: The size of the patch.

    Returns:
        The output tensor with shape (..., H * P, W * P, C).
    """
    assert height % patch_size == 0, "Expected height to be divisible by patch_size."
    assert width % patch_size == 0, "Expected width to be divisible by patch_size."

    x = rearrange(
        x,
        "... (h w) (ph pw c) -> ... (h ph) (w pw) c",
        h=height // patch_size,
        w=width // patch_size,
        ph=patch_size,
        pw=patch_size,
    )
    return x






def timeit(repeats: int, f: Callable, *args, **kwargs) -> float:
    torch.cuda.reset_peak_memory_stats()
    mem_tic = torch.cuda.max_memory_allocated() / 1024**3

    for _ in range(5):  # warmup
        f(*args, **kwargs)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        results = f(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    mem = torch.cuda.max_memory_allocated() / 1024**3 - mem_tic
    return (end - start) / repeats, mem, results
