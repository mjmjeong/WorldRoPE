"""Map token features to the depth parameters used to construct RayRoPE rays.

Outputs are raw log-depth / interval-width parameters, not Gaussian support
parameters. Conversion to depth intervals and ray lifting stay in attention.py.
"""

import torch
import torch.nn.functional as F
from torch.nn import Parameter

############################################################################
# Register the original token-wise log-depth / interval-width head on the
# MHA module.
############################################################################
def initialize_depth_head(self, embed_dim, predict_d, init_depth, init_sigma):
    if predict_d == "predict_dsig":
        self.d_proj_weight = Parameter(torch.zeros((2, embed_dim)))
        self.d_proj_bias = Parameter(torch.tensor([init_depth, init_sigma]))
    elif predict_d == "predict_d":
        self.d_proj_weight = Parameter(torch.zeros((1, embed_dim)))
        self.d_proj_bias = Parameter(torch.tensor([init_depth]))
    elif predict_d == "none":
        self.register_parameter("d_proj_weight", None)
        self.register_parameter("d_proj_bias", None)
    else:
        raise ValueError(f"Unsupported predict_d mode: {predict_d}")


############################################################################
# Predict raw ray-depth parameters from token features before Q/K/V projection.
# The last dimension matches the registered depth head, including per-head modes.
# Return None when depth prediction is disabled; interval decoding is downstream.
############################################################################
def predict_depth_parameters(self, query):
    if self.predict_d == "none":
        return None
    return F.linear(query, self.d_proj_weight, self.d_proj_bias)
