#!/usr/bin/env python3
"""
PROBE sequence scoring entry point (no generation).

Score pre-generated sequences with the same ensemble pairwise fitness used in
``generate.py``. Reports mean fitness and uncertainty per model type, and for
the full ensemble.
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.models import EnsembleModel
from src.data.data_preprocessing import DataPreprocessor
from src.data.dataloader import sequence_to_indices, _pad_seqs
from src.utils.utils import (
    print_all_args,
    load_data_config_from_yaml,
    load_generate_config_from_yaml,
    resolve_config_path,
    resolve_data_identity,
    resolve_checkpoint_dir,
)

SEQ_COL_CANDIDATES = ('seq', 'gen_seq', 'sequence', 'generated_seq')
REF_COL_CANDIDATES = ('ref_seq', 'reference_seq', 'reference', 'ref')


def _pick_column(columns, candidates: Tuple[str, ...]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def load_sequences(seq_file: str) -> Tuple[List[str], Optional[List[str]]]:
    """Load generated sequences (and optional per-row reference sequences)."""
    df = pd.read_csv(seq_file)
    if df.empty:
        raise ValueError(f"Empty sequence file: {seq_file}")

    seq_col = _pick_column(df.columns, SEQ_COL_CANDIDATES)
    if seq_col is None:
        # Single unnamed / first column fallback when header is missing.
        if len(df.columns) == 1:
            seq_col = df.columns[0]
        else:
            raise ValueError(
                f"Cannot find sequence column in {list(df.columns)}; "
                f"expected one of {SEQ_COL_CANDIDATES}"
            )

    ref_col = _pick_column(df.columns, REF_COL_CANDIDATES)
    sequences = df[seq_col].astype(str).tolist()
    ref_seqs = df[ref_col].astype(str).tolist() if ref_col is not None else None
    return sequences, ref_seqs


def build_ref_pool(args, preprocessor: DataPreprocessor) -> List[str]:
    """Top-k enriched sequences used when no per-row ref_seq is provided."""
    if args.reference_sequences is not None:
        refs = args.reference_sequences
        if isinstance(refs, str):
            refs = [refs]
        print(f"Use the provided reference sequence(s) as pool ({len(refs)})")
        return list(refs)

    top_k = args.top_k
    ref_df = preprocessor.find_best_sequences(
        top_k=top_k, seq_length=args.seq_length)
    pool = ref_df['seq'].tolist()
    print(f"Use the top {len(pool)} enriched sequences as reference pool")
    return pool


def _encode_refs(refs: List[str], device: str) -> torch.Tensor:
    idx_list = [sequence_to_indices(seq) for seq in refs]
    return _pad_seqs(idx_list).to(device)


def summarize_rows(
    rows: List[Dict],
    model_types: List[str],
) -> Dict[str, Dict[str, float]]:
    """Average per-sequence score / uncertainty over the batch."""
    summary: Dict[str, Dict[str, float]] = {}

    ens_scores = np.asarray([r['score'] for r in rows], dtype=np.float64)
    ens_uncs = np.asarray([r['uncertainty'] for r in rows], dtype=np.float64)
    summary['ensemble'] = {
        'mean_score': float(ens_scores.mean()),
        'std_score': float(ens_scores.std()) if len(ens_scores) > 1 else 0.0,
        'mean_uncertainty': float(ens_uncs.mean()),
        'std_uncertainty': float(ens_uncs.std()) if len(ens_uncs) > 1 else 0.0,
        'n_sequences': float(len(rows)),
    }

    for mtype in model_types:
        scores = np.asarray(
            [r['by_type'][mtype]['score'] for r in rows], dtype=np.float64)
        uncs = np.asarray(
            [r['by_type'][mtype]['uncertainty'] for r in rows], dtype=np.float64)
        summary[mtype] = {
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std()) if len(scores) > 1 else 0.0,
            'mean_uncertainty': float(uncs.mean()),
            'std_uncertainty': float(uncs.std()) if len(uncs) > 1 else 0.0,
            'n_sequences': float(len(rows)),
        }
    return summary


def print_summary(summary: Dict[str, Dict[str, float]], model_types: List[str]):
    print(f"\n{'=' * 60}")
    print("Scoring summary")
    print(f"{'=' * 60}")
    order = list(model_types) + ['ensemble']
    for key in order:
        s = summary[key]
        print(
            f"  {key:20s}  "
            f"mean={s['mean_score']:.4f}±{s['std_score']:.4f}  "
            f"unc={s['mean_uncertainty']:.4f}±{s['std_uncertainty']:.4f}"
        )
    print(f"{'=' * 60}")


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

    sequences, ref_seqs = load_sequences(args.seq_file)
    print(f"Loaded {len(sequences)} sequences from {args.seq_file}")
    if ref_seqs is None:
        print("No ref_seq column found → score against reference pool")
    else:
        print("ref_seq column found → score each sequence against its own ref")

    model = EnsembleModel(
        model_name=args.model_name,
        ckpt_dir=args.ckpt_dir,
        device=device,
        vocab_size=args.vocab_size,
        score_temperature=args.score_temperature,
    )

    preprocessor = DataPreprocessor(
        data_path=args.data_path,
        data_name=args.data_name,
        data_output_dir=args.data_output_dir,
        load_raw_data=True,
    )

    pool_refs = None
    pool_indices = None
    if ref_seqs is None:
        pool_refs = build_ref_pool(args, preprocessor)
        pool_indices = _encode_refs(pool_refs, device)

    rows: List[Dict] = []
    for i, seq in enumerate(tqdm(sequences, desc='Scoring')):
        if ref_seqs is None:
            ref_indices = pool_indices
            ref_used = None
        else:
            ref_used = ref_seqs[i]
            ref_indices = _encode_refs([ref_used], device)

        result = model.score_sequence(seq, ref_indices)
        row = {
            'seq': seq,
            'score': result['score'],
            'uncertainty': result['uncertainty'],
            'by_type': result['by_type'],
            'fold_scores': result['fold_scores'],
        }
        if ref_used is not None:
            row['ref_seq'] = ref_used
        rows.append(row)

    summary = summarize_rows(rows, model.model_types)
    print_summary(summary, model.model_types)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.seq_file).stem

    # Per-sequence CSV
    flat_rows = []
    for r in rows:
        item = {
            'seq': r['seq'],
            'score': r['score'],
            'uncertainty': r['uncertainty'],
        }
        if 'ref_seq' in r:
            item['ref_seq'] = r['ref_seq']
        for mtype, stats in r['by_type'].items():
            item[f'{mtype}_score'] = stats['score']
            item[f'{mtype}_uncertainty'] = stats['uncertainty']
        flat_rows.append(item)

    seq_csv = out_dir / f'{stem}_scores.csv'
    pd.DataFrame(flat_rows).to_csv(seq_csv, index=False)
    print(f"Per-sequence scores saved: {seq_csv}")

    # Summary JSON
    summary_json = out_dir / f'{stem}_score_summary.json'
    payload = {
        'seq_file': str(args.seq_file),
        'data_name': args.data_name,
        'model_name': list(args.model_name),
        'ckpt_dir': args.ckpt_dir,
        'n_sequences': len(sequences),
        'ref_mode': 'per_row' if ref_seqs is not None else 'pool',
        'n_pool_refs': None if pool_refs is None else len(pool_refs),
        'summary': summary,
    }
    with open(summary_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Summary saved: {summary_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PROBE: score generated DNA aptamer sequences')
    parser.add_argument('--seq_file', type=str, required=True,
                        help='CSV of generated sequences '
                             '(seq[,ref_seq] or gen_seq[,ref_seq])')
    parser.add_argument(
        '--data_path', type=str, default=None,
        help='Path to seq/value CSV (data_name defaults to file stem)')
    parser.add_argument(
        '--data_name', type=str, default=None,
        help='Dataset name; defaults to CSV stem')
    parser.add_argument('--model_name', nargs='+', default=None,
                        help='Encoder name(s), e.g. cnn_lstm cnn_transformer')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for score CSV / summary JSON')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Reference pool size when no ref_seq column')
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
        'cli': vars(cli),
    }

    args.seq_file = cli.seq_file

    if cli.model_name is not None:
        args.model_name = cli.model_name
        config['generate']['model_name'] = cli.model_name

    if cli.ckpt_dir is not None:
        args.ckpt_dir = cli.ckpt_dir
        config['generate']['ckpt_dir'] = cli.ckpt_dir

    if cli.top_k is not None:
        args.top_k = cli.top_k
        config['generate']['top_k'] = cli.top_k

    if cli.output_dir is not None:
        args.output_dir = cli.output_dir
    else:
        args.output_dir = f'experiments/scoring/{args.data_name}/'

    main(args, config)
