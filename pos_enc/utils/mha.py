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

from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.parameter import Parameter
from pos_enc.worldrope.token_with_gaussian_splatting import initialize_gmm_heads, predict_gmm_parameters
from pos_enc.rayrope.token_to_ray import initialize_depth_head, predict_depth_parameters


# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)


class MultiheadAttention(torch.nn.Module):
    """
    Same as torch.nn.MultiheadAttention for bidirectional attention, but supports:
    - RMSNorm
    - customized attention function
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        predict_d: str = "none",
        init_depth: float = 0.0,
        init_sigma: float = 3.0,
        predict_anchor_weights: int = 0,
        predict_anchor_sigmas: int = 0,
        predict_anchor_means: int = 0,
        anchor_mean_init: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.bias = bias
        self.qk_norm = qk_norm
        self.predict_d = predict_d
        self.predict_anchor_weights = predict_anchor_weights
        self.predict_anchor_sigmas = predict_anchor_sigmas
        self.predict_anchor_means = predict_anchor_means

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))

        if self.bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        # Vanilla RayRoPE predicts one shared depth interval per token and
        # Transformer layer, before the Q/K/V projections.  The two outputs
        # are raw log-depth and raw interval half-width respectively.
        initialize_depth_head(self, embed_dim, predict_d, init_depth, init_sigma)

        # Learned URoPE anchor mixture.  Like RayRoPE's depth head, this is
        # predicted from the shared pre-QKV token feature independently in
        # every Transformer layer.  Zero initialization gives an exact
        # uniform simplex at step zero and consumes no model-init RNG.
        initialize_gmm_heads(self, embed_dim, predict_anchor_weights, predict_anchor_sigmas, predict_anchor_means, anchor_mean_init)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        sdpa_fn: Callable = F.scaled_dot_product_attention,
    ) -> Tensor:
        """
        Args:
            query: (B, T, D)
            key: (B, T, D)
            value: (B, T, D)
            sdpa_fn: (q, k, v, **kwargs) -> Tensor
        Returns:
            output: (B, T, D)
        """
        predicted_d = predict_depth_parameters(self, query)
        anchor_weights, anchor_sigmas, anchor_means = predict_gmm_parameters(self, query)

        q_proj_weight, k_proj_weight, v_proj_weight = self.in_proj_weight.chunk(
            3, dim=0
        )
        if self.in_proj_bias is not None:
            q_proj_bias, k_proj_bias, v_proj_bias = self.in_proj_bias.chunk(3, dim=0)
        else:
            q_proj_bias = k_proj_bias = v_proj_bias = None

        q = F.linear(query, q_proj_weight, q_proj_bias)
        k = F.linear(key, k_proj_weight, k_proj_bias)
        v = F.linear(value, v_proj_weight, v_proj_bias)
        q, k, v = (
            rearrange(x, "b t (h c) -> b t h c", h=self.num_heads) for x in [q, k, v]
        )
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k, v = (rearrange(x, "b t h c -> b h t c") for x in [q, k, v])
        sdpa_kwargs = {"dropout_p": self.dropout if self.training else 0.0}
        if predicted_d is not None:
            sdpa_kwargs["predicted_d"] = predicted_d
        if anchor_weights is not None:
            sdpa_kwargs["anchor_weights"] = anchor_weights
        if anchor_sigmas is not None:
            sdpa_kwargs["anchor_sigmas"] = anchor_sigmas
        if anchor_means is not None:
            sdpa_kwargs["anchor_means"] = anchor_means
        o = sdpa_fn(q, k, v, **sdpa_kwargs)
        if isinstance(o, tuple):
            o, attn_scores = o
        else:
            attn_scores = None
        o = rearrange(o, "b h t c -> b t (h c)", h=self.num_heads)
        attn_output = F.linear(o, self.out_proj.weight, self.out_proj.bias)
        if attn_scores is not None:
            return attn_output, attn_scores
        else:
            return attn_output


if __name__ == "__main__":
    torch.manual_seed(42)

    device = "cuda:0"
    mha = MultiheadAttention(embed_dim=32, num_heads=8).to(device)

    q = torch.randn(10, 16, 32).to(device)
    k = torch.randn(20, 16, 32).to(device)
    v = torch.randn(20, 16, 32).to(device)

    output = mha(q, k, v, sdpa_fn=lambda q, k, v, **kwargs: v)
    print(output.size(), output.sum())
