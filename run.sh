#!/usr/bin/env bash
# Optional helper — edit DATA_PATH / uncomment steps as needed.
set -euo pipefail
cd "$(dirname "$0")"

DATA_PATH=notebooks/demo.csv

# python scripts/data.py --data_path "$DATA_PATH"
# python scripts/train.py --data_path "$DATA_PATH" --models cnn
python scripts/generate.py --data_path "$DATA_PATH" --model_name cnn
