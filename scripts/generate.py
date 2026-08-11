#!/usr/bin/env python3
"""
PROBE sequence generation entry point (gradient ascent).
"""

import sys
import argparse
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.models import EnsembleModel
from src.generation.generator import PROBEGenerator
from src.data.data_preprocessing import DataPreprocessor
from src.utils.utils import (
    print_all_args,
    load_data_config_from_yaml,
    load_generate_config_from_yaml,
    resolve_config_path,
    resolve_data_identity,
    resolve_checkpoint_dir,
)


def main(args, config):
    print_all_args(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if not args.ckpt_dir:
        args.ckpt_dir = str(resolve_checkpoint_dir(
            data_name=args.data_name,
            split_method=args.split_method,
            max_n_seqs=args.max_n_seqs,
            mode=getattr(args, 'mode', 'pairwise'),
            local_pairing=getattr(args, 'local_pairing', True),
        ))
        print(f"ckpt_dir not set → using {args.ckpt_dir}")

    model = EnsembleModel(
        model_name=args.model_name,
        ckpt_dir=args.ckpt_dir,
        device=device,
        vocab_size=args.vocab_size,
        score_temperature=args.score_temperature,
    )
    print(
        f"Loaded {len(model.models)} ensemble members; "
        f"generation_model_index={args.generation_model_index}")

    preprocessor = DataPreprocessor(
        data_path=args.data_path,
        data_name=args.data_name,
        data_output_dir=args.data_output_dir,
        load_raw_data=True,
    )

    if args.reference_sequences is None:
        # Larger pool for freq_prior statistics; fitness uses top_k of these.
        full_k = max(int(args.top_k), int(args.num_sequences))
        ref_df = preprocessor.find_best_sequences(
            top_k=full_k, seq_length=args.seq_length)
        ref_pool = ref_df['seq'].tolist()
        print(
            f"Reference pool: top {len(ref_pool)} enriched sequences "
            f"(fitness uses top_k={args.top_k})"
        )
    else:
        refs = args.reference_sequences
        ref_pool = refs if isinstance(refs, (list, tuple)) else [refs]
        print(f"Use the provided reference sequence(s) as reference ({len(ref_pool)})")

    init_type = args.init_type
    if init_type == 'ref_content':
        init_type = 'freq_prior'
        print("Note: init_type 'ref_content' renamed to 'freq_prior'")

    print(f"\n{'=' * 60}")
    print("Starting generation process")
    print(f"{'=' * 60}")
    print(f"Init Type: {init_type}")
    print(f"Beta: {args.beta}")
    print(f"Alpha: {args.alpha}")

    model_names = args.model_name if isinstance(args.model_name, (list, tuple)) else [args.model_name]
    all_models = '-'.join(model_names)
    generate_dir = (
        f'{args.generate_dir}/{args.data_name}_{all_models}/'
        f'{init_type}_{args.num_sequences}'
    )

    generator = PROBEGenerator(
        model=model,
        reference_pool=ref_pool,
        seq_length=args.seq_length,
        skip_t_threshold=args.skip_t_threshold,
        top_k=args.top_k,
        generate_dir=generate_dir,
        device=device,
    )

    seqs = generator.generate(
        init_type=init_type,
        num_sequences=args.num_sequences,
        num_iterations=args.num_iterations,
        learning_rate=args.refine_learning_rate,
        temperature=args.temperature,
        beta=args.beta,
        alpha=args.alpha,
        improve_patience=args.improve_patience,
        freeze_patience=args.freeze_patience,
        generation_model_index=args.generation_model_index,
    )

    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"{'=' * 60}")

    print(f"\n{'=' * 60}")
    print("Top 10 sequences:")
    print(seqs.head(10))
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PROBE: generate optimized DNA aptamer sequences')
    parser.add_argument(
        '--data_path', type=str, default=None,
        help='Path to seq/value CSV (data_name defaults to file stem)')
    parser.add_argument(
        '--data_name', type=str, default=None,
        help='Dataset name for outputs; defaults to CSV stem')
    parser.add_argument('--model_name', nargs='+', default=None,
                        help='Encoder name(s), e.g. cnn transformer')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint directory (overrides YAML / auto path)')
    parser.add_argument(
        '--init_type', type=str, default=None,
        choices=['topk', 'random', 'freq_prior', 'ref_content'],
        help="Initialization strategy (ref_content is a legacy alias of freq_prior)")
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

    generate_cfg = resolve_config_path('generate', data_name)
    generate_args, generate_config = load_generate_config_from_yaml(generate_cfg)

    merged = {
        **vars(data_args),
        **vars(generate_args),
    }
    args = SimpleNamespace(**merged)
    config = {
        'data_config_path': str(data_cfg),
        'generate_config_path': str(generate_cfg),
        'data': vars(data_args),
        'generate': vars(generate_args),
    }

    if cli.model_name is not None:
        args.model_name = cli.model_name
        config['generate']['model_name'] = cli.model_name

    if cli.ckpt_dir is not None:
        args.ckpt_dir = cli.ckpt_dir
        config['generate']['ckpt_dir'] = cli.ckpt_dir

    if cli.init_type is not None:
        args.init_type = 'freq_prior' if cli.init_type == 'ref_content' else cli.init_type
        config['generate']['init_type'] = args.init_type

    main(args, config)
