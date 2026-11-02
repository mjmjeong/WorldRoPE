"""Model recipes corresponding to the two preserved paper source snapshots."""
from nvs.lvsm import LVSMDecoderOnlyModelConfig

def model_config(method="worldrope", dataset="re10k", resolution=256):
    if method not in {"worldrope", "rayrope", "urope", "plucker", "prope", "xyrope"}:
        raise ValueError(f"Unknown method: {method}")
    if dataset not in {"re10k", "objaverse"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    config = LVSMDecoderOnlyModelConfig(ref_views=2, tar_views=1)
    config.img_shape = (resolution, resolution, 3)
    config.cam_shape = (resolution, resolution, 6)
    config.patch_size = 8
    config.encoder.num_layers = 6
    config.encoder.layer.d_model = 768
    config.encoder.layer.nhead = 16
    config.encoder.layer.dim_feedforward = 1024
    config.encoder.layer.qk_norm = True
    config.ray_encoding = "plucker" if method == "plucker" else "camray"
    config.pos_enc = {"worldrope": "urope_uvd_axial", "rayrope": "rayrope", "urope": "urope", "plucker": "none", "prope": "prope", "xyrope": "rope2d"}[method]
    if method == "worldrope":
        config.urope_learned_anchor_gaussian_mixture = True
        config.urope_learned_anchor_gmm_learned_means = True
        config.urope_fixed_lateral_footprint_kv = True
    config.urope_min_depth, config.urope_max_depth = (2.0, 20.0) if dataset == "re10k" else (0.7, 2.6)
    return config
