"""WorldRoPE with Gaussian support and signed-depth axial defaults."""
from pos_enc.utils.attention import MultiViewRoPEAttention

class WorldRoPEDotProductAttention(MultiViewRoPEAttention):
    ############################################################################
    # Configure the WorldRoPE path on the shared multi-view attention wrapper.
    # Enable learned Gaussian supports, patch footprint and signed-depth axial RoPE.
    # Explicit keyword arguments may select the preserved mixed-frequency variants.
    ############################################################################
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("coordinate_mode", "uvd_axial")
        kwargs.setdefault("learned_anchor_mixture", True)
        kwargs.setdefault("learned_anchor_gaussian_mixture", True)
        kwargs.setdefault("learned_anchor_means", True)
        kwargs.setdefault("fixed_lateral_footprint_kv", True)
        super().__init__(*args, **kwargs)

    ########################################################################
    # Build the shared MHA with Gaussian-head defaults matching this encoding.
    # Derive the initial means from the configured depth range and component count.
    # Keep parameter registration unchanged so saved checkpoint keys still match.
    ########################################################################
    def make_mha(self, embed_dim: int, **kwargs):
        from pos_enc.utils.mha import MultiheadAttention
        if embed_dim != self.head_num * self.head_dim:
            raise ValueError("embed_dim must equal head_num * head_dim")
        components = self.depth_num
        bin_size = (self.max_depth - self.min_depth) / components
        initial_means = tuple(
            self.min_depth + bin_size * index
            for index in range(components)
        )
        kwargs.setdefault("predict_anchor_weights", components if self.learned_anchor_mixture else 0)
        kwargs.setdefault("predict_anchor_sigmas", components if self.learned_anchor_gaussian_mixture else 0)
        kwargs.setdefault("predict_anchor_means", components if self.learned_anchor_means else 0)
        kwargs.setdefault("anchor_mean_init", initial_means)
        return MultiheadAttention(embed_dim=embed_dim, num_heads=self.head_num, **kwargs)
