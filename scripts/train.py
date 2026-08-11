#!/usr/bin/env python3
"""
cross-validation training script.
"""

import sys
import argparse
import warnings
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.utils import (
    Tee,
    print_all_args,
    load_data_config_from_yaml,
    load_train_config_from_yaml,
    resolve_config_path,
    resolve_data_identity,
)
from src.utils.config import MODEL_CONFIG
from src.data.data_preprocessing import DataPreprocessor
from src.models.trainers import CVTrainer


def main(args, config):

    print(
        f'\n=== Loading dataset: {args.data_name} '
        f'(source={args.data_path}) {args.split_method} '
        f'{args.n_splits} folds ===\n'
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

    all_folds_data = preprocessor.load_all_folds_data(local_pairing=args.local_pairing)
    if len(all_folds_data) != args.n_splits:
        raise ValueError(f"Expected {args.n_splits} folds, got {len(all_folds_data)}")
    preprocessor._load_fold_assignments()

    folds_seqs = preprocessor.fold_assignments.merge(preprocessor.df, on='seq', how='left')

    trainer = CVTrainer(
        data_name=args.data_name,
        mode=args.mode,
        model_names=args.models,
        folds_pairs=all_folds_data,
        folds_seqs=folds_seqs,
        output_dir=f'{args.output_dir}/{args.data_name}',
        local_pairing=args.local_pairing,
        random_state=args.random_state,
        args=args,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path('logs') / (
        f"{args.data_name}/{args.split_method}_{args.max_n_seqs}_"
        f"{args.mode}_{timestamp}.txt"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    tee = Tee(log_file, mode='w')

    try:
        print(f"Log: {tee.file.name}")
        print(f"Start: {timestamp}")
        print_all_args(config)
        trainer.train_cv()
        print(f"\nDone! Results: {trainer.output_dir}")
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        tee.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross-validation training')
    parser.add_argument(
        '--data_path', type=str, default=None,
        help='Path to seq/value CSV (data_name defaults to file stem)')
    parser.add_argument(
        '--data_name', type=str, default=None,
        help='Dataset name for outputs/checkpoints; defaults to CSV stem')
    parser.add_argument('--mode', type=str, default=None, choices=['pairwise', 'pointwise'])
    parser.add_argument('--gpu_id', type=str, default=None)
    parser.add_argument('--models', type=str, nargs='+', choices=MODEL_CONFIG,
                        help='backbone model name(s)')
    parser.add_argument('--lr', default=None, nargs='+', help='Learning rate(s)')
    parser.add_argument('--no-local', action='store_true')
    parser.add_argument('--debug', action='store_true')
    cli = parser.parse_args()

    peek_name = cli.data_name or (Path(cli.data_path).stem if cli.data_path else None)
    data_cfg = resolve_config_path('data', peek_name)
    data_args = load_data_config_from_yaml(data_cfg)
    if cli.data_path is not None:
        data_args.data_path = cli.data_path
    if cli.data_name is not None:
        data_args.data_name = cli.data_name

    data_path, data_name = resolve_data_identity(
        data_path=data_args.data_path,
        data_name=data_args.data_name,
        data_output_dir=data_args.data_output_dir,
    )
    data_args.data_path = str(data_path)
    data_args.data_name = data_name

    train_cfg = resolve_config_path('train', data_name)
    train_args, model_args = load_train_config_from_yaml(train_cfg)

    merged = {**vars(data_args), **vars(train_args), **vars(model_args)}
    args = SimpleNamespace(**merged)
    config = {
        'data_config_path': str(data_cfg),
        'train_config_path': str(train_cfg),
        'data': vars(data_args),
        'train': vars(train_args),
        'model': vars(model_args),
    }

    if cli.mode is not None:
        args.mode = cli.mode
        config['train']['mode'] = cli.mode

    if cli.gpu_id is not None:
        args.gpu_id = cli.gpu_id
        config['train']['gpu_id'] = cli.gpu_id

    if cli.models is not None:
        args.models = cli.models
        config['model']['models'] = cli.models

    if cli.lr is not None:
        args.learning_rate = cli.lr
        config['train']['learning_rate'] = cli.lr

    if cli.no_local:
        args.local_pairing = False
        config['train']['local_pairing'] = False

    args.debug = cli.debug
    config['train']['debug'] = cli.debug

    main(args, config)
