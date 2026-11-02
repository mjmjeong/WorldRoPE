"""Fixed-anchor URoPE API over the shared multi-view camera folding/SDPA wrapper.

The baseline has no learned Gaussian head. Its projection geometry is in
multi_head_anchor.py; the shared wrapper also serves WorldRoPE with different options.
"""

from pos_enc.utils.attention import (
    MultiViewRoPEAttention as URoPEDotProductAttention,
    multiview_rope_dot_product_attention as urope_dot_product_attention,
)
