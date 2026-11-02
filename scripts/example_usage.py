#!/usr/bin/env python3
"""Standalone WorldRoPE attention example; synthetic cameras, CPU, no assets."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from pos_enc.worldrope import WorldRoPEDotProductAttention

def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    cameras, patches_x, patches_y = 3, 2, 2
    attention = WorldRoPEDotProductAttention(head_num=16, head_dim=48,
        patches_x=patches_x, patches_y=patches_y, image_width=16, image_height=16)
    mha = attention.make_mha(embed_dim=768)
    x = torch.randn(1, cameras * patches_x * patches_y, 768, requires_grad=True)
    viewmats = torch.eye(4).repeat(1, cameras, 1, 1)
    viewmats[0, :, 0, 3] = torch.tensor([0.0, 0.25, -0.25])
    Ks = torch.tensor([[24., 0., 8.], [0., 24., 8.], [0., 0., 1.]]).repeat(1, cameras, 1, 1)
    def sdpa(q, k, v, **kwargs):
        return attention(q, k, v, viewmats=viewmats, Ks=Ks, **kwargs)
    y = mha(x, x, x, sdpa_fn=sdpa)
    y.square().mean().backward()
    assert y.shape == x.shape and torch.isfinite(y).all()
    for name in ["anchor_proj_weight", "anchor_sigma_proj_weight", "anchor_mean_proj_weight"]:
        grad = getattr(mha, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    print("WorldRoPE output:", tuple(y.shape), "; mean/sigma/weight gradients: finite and nonzero")

if __name__ == "__main__":
    main()
