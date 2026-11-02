"""Read legacy tagged recipes without importing or executing YAML object tags."""
from dataclasses import fields
from pathlib import Path
import yaml
import torch.nn.functional as F
from nvs.lvsm import LVSMDecoderOnlyModelConfig
from pos_enc.utils.transformer import TransformerEncoderConfig, TransformerEncoderLayerConfig

class RecipeLoader(yaml.SafeLoader):
    pass

def _object(loader, suffix, node):
    allowed = {"__main__.LVSMLauncherConfig", "nvs.trainval.LVSMLauncherConfig", "nvs.lvsm.LVSMDecoderOnlyModelConfig"}
    allowed.update(prefix + name for prefix in ["urope.utils.transformer.", "pos_enc.utils.transformer."] for name in ["TransformerEncoderConfig", "TransformerEncoderLayerConfig"])
    if suffix not in allowed:
        raise ValueError(f"Unsupported recipe object tag: {suffix}")
    return loader.construct_mapping(node, deep=True)

def _name(loader, suffix, node):
    allowed = {"torch.nn.functional.relu"}
    allowed.update(prefix + name for prefix in ["urope.utils.transformer.", "pos_enc.utils.transformer."] for name in ["TransformerEncoder", "TransformerEncoderLayer"])
    if suffix not in allowed:
        raise ValueError(f"Unsupported recipe callable: {suffix}")
    return suffix

RecipeLoader.add_multi_constructor("tag:yaml.org,2002:python/object:", _object)
RecipeLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _name)
RecipeLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node, deep=True)))

def read_recipe(path):
    data = yaml.load(Path(path).read_text(), Loader=RecipeLoader)
    if not isinstance(data, dict) or "model_config" not in data:
        raise ValueError("Expected a mapping containing model_config")
    return data

def _construct(cls, data):
    data = dict(data)
    data.pop("_target", None)
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**data)

def model_from_recipe(data):
    model = dict(data["model_config"])
    encoder = dict(model["encoder"])
    layer = dict(encoder["layer"])
    activation = layer.get("activation", "relu")
    if activation not in ["relu", "torch.nn.functional.relu"]:
        raise ValueError(f"Unsupported activation: {activation}")
    layer["activation"] = F.relu
    encoder["layer"] = _construct(TransformerEncoderLayerConfig, layer)
    model["encoder"] = _construct(TransformerEncoderConfig, encoder)
    for key in ["img_shape", "cam_shape"]:
        if key in model:
            model[key] = tuple(model[key])
    return _construct(LVSMDecoderOnlyModelConfig, model)

def portable_recipe(data):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k != "_target"}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    data = clean(data)
    for key in ["re10k_fast_train_dir", "re10k_legacy_schedule_file", "re10k_image_cache_dir", "objaverse_fast_dir", "objaverse_schedule_file", "resume"]:
        if key in data:
            data[key] = None
    data["output_dir"] = "results/worldrope"
    return data
