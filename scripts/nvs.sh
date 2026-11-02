#!/usr/bin/env bash
set -euo pipefail
release_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$release_script_dir/launch_nvs.py" "$@"
