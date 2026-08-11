#!/usr/bin/env python3
"""Build CV fold pair CSVs from a seq/value CSV and data_config.yaml."""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_preprocessing import DataPreprocessor
from src.utils.utils import (
    load_data_config_from_yaml,
    resolve_config_path,
    resolve_data_identity,
)


def main(args):
    
    print(
        f'\n=== {args.data_name} data preprocessing started '
        f'(data_path={args.data_path}) ===\n'
    )

    preprocessor = DataPreprocessor(
        data_path=args.data_path,
        data_name=args.data_name,
        data_output_dir=args.data_output_dir,
        random_state=args.random_state,
        n_splits=args.n_splits,
        split_method=args.split_method,
        max_n_seqs=args.max_n_seqs,
        vocab_size=args.vocab_size,
        load_raw_data=True,
        min_value=args.min_value,
        log_transform=args.log_transform,
    )
    preprocessor.create_pairs(val_size=args.val_size)
    print(f'\n=== {preprocessor.data_name} preprocessing done ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing (config-driven)')
    parser.add_argument(
        '--data_path', type=str, default=None,
        help='Path to seq/value CSV (data_name defaults to file stem if omitted)')
    parser.add_argument(
        '--data_name', type=str, default=None,
        help='Dataset name for outputs/checkpoints; defaults to CSV stem')
    cli = parser.parse_args()

    # Peek at stem for named config lookup when only --data_path is given.
    peek_name = cli.data_name
    if peek_name is None and cli.data_path:
        peek_name = Path(cli.data_path).stem

    config_path = resolve_config_path('data', peek_name)
    args = load_data_config_from_yaml(config_path)

    # CLI overrides YAML.
    if cli.data_path is not None:
        args.data_path = cli.data_path
    if cli.data_name is not None:
        args.data_name = cli.data_name

    data_path, data_name = resolve_data_identity(
        data_path=args.data_path,
        data_name=args.data_name,
        data_output_dir=args.data_output_dir,
    )
    args.data_path = str(data_path)
    args.data_name = data_name
    main(args)
