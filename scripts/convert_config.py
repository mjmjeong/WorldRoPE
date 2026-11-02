#!/usr/bin/env python3
"""Convert a legacy tagged config into portable, plain YAML; weights stay unchanged."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from nvs.config_io import read_recipe, model_from_recipe, portable_recipe

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    data = read_recipe(args.source)
    model_from_recipe(data)
    data = portable_recipe(data)
    model_from_recipe(data)
    with args.destination.open("x") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    print(args.destination)

if __name__ == "__main__":
    main()
