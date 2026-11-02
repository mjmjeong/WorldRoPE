#!/usr/bin/env python3
"""Strict-load a saved WorldRoPE checkpoint with its saved or converted config."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from nvs.config_io import read_recipe, model_from_recipe
    from nvs.lvsm import LVSMDecoderOnlyModel
    torch.set_num_threads(1)
    model = LVSMDecoderOnlyModel(model_from_recipe(read_recipe(args.config)))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = payload["model"]
    result = model.load_state_dict(state, strict=True)
    print(json.dumps(dict(strict_load=True, tensor_count=len(state), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), step=payload.get("step"), missing_keys=result.missing_keys, unexpected_keys=result.unexpected_keys, checkpoint_conversion_required=False), indent=2))

if __name__ == "__main__":
    main()
