import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Optional, Tuple
import yaml
from src.utils.config import GENERATION_DIR

project_root = Path(__file__).parent.parent.parent


def resolve_data_identity(
    data_path: Optional[str] = None,
    data_name: Optional[str] = None,
    data_output_dir: str = './dataset',
    raw_data_dir: Optional[str] = None,
) -> Tuple[Path, str]:
    """Resolve raw CSV path and dataset name.

    Priority:
      1. ``data_path`` – load this file; ``data_name`` defaults to its stem
      2. ``data_name`` – load ``{raw_data_dir or data_output_dir}/{data_name}.csv``
    """
    if data_path:
        path = Path(data_path).expanduser()
        if not path.is_absolute():
            # Prefer CWD-relative (scripts run from project root), then project root.
            cwd_candidate = Path.cwd() / path
            root_candidate = project_root / path
            path = cwd_candidate if cwd_candidate.exists() else root_candidate
        path = path.resolve()
        name = data_name or path.stem
        return path, str(name)

    if data_name:
        root = Path(raw_data_dir or data_output_dir)
        path = root / f'{data_name}.csv'
        if not path.is_absolute():
            cwd_candidate = Path.cwd() / path
            root_candidate = project_root / path
            path = cwd_candidate if cwd_candidate.exists() else root_candidate
        return path.resolve(), str(data_name)

    raise ValueError(
        'Provide --data_path (preferred) or --data_name. '
        'If only a CSV path is given, data_name defaults to the file stem.'
    )


def resolve_config_path(kind: str, data_name: Optional[str] = None) -> Path:
    """Return ``config/{data_name}/{kind}_config.yaml`` or the shared fallback."""
    if data_name:
        named = Path(f'config/{data_name}/{kind}_config.yaml')
        if named.exists():
            return named
    fallback = Path(f'config/{kind}_config.yaml')
    if not fallback.exists():
        raise FileNotFoundError(f'Config not found for kind={kind!r}: {fallback}')
    return fallback


def resolve_checkpoint_dir(
    data_name: str,
    split_method: str,
    max_n_seqs,
    mode: str = 'pairwise',
    local_pairing: bool = True,
) -> Path:
    """Directory layout matching ``resolve_checkpoint_path`` in trainers."""
    base = Path(f'checkpoints/{data_name}/{split_method}')
    if mode == 'pointwise':
        return base / 'pointwise'
    if not local_pairing:
        return base / f'global_{float(max_n_seqs)}'
    return base / str(float(max_n_seqs))


def resolve_checkpoint_path(
    data_name,
    split_method,
    max_n_seqs,
    model_name,
    lr,
    fold_id,
    mode='pairwise',
    local_pairing=True,
):
    """Checkpoint path layout used by training and evaluation."""
    lr_tag = str(lr).replace('.', 'p')
    ckp_name = f"{model_name}_fold{fold_id}_lr{lr_tag}.pt"
    base = Path(f'checkpoints/{data_name}/{split_method}')
    if mode == 'pointwise':
        return base / 'pointwise' / ckp_name
    if not local_pairing:
        return base / f'global_{float(max_n_seqs)}' / ckp_name
    return base / str(float(max_n_seqs)) / ckp_name


class Tee:
    """Tee stdout to both console and a log file."""
    def __init__(self, file_path, mode='a'):
        self.file = open(file_path, mode, encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()

def load_data_config_from_yaml(config_path):
    """Load the `data` section from YAML into a SimpleNamespace."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        data_config = (yaml.safe_load(f) or {}).get('data', {}) or {}

    args = SimpleNamespace(
        data_path=data_config.get('data_path', None),
        data_name=data_config.get('data_name', None),
        data_output_dir=data_config.get('data_output_dir', './dataset'),
        random_state=data_config.get('random_state', 42),
        val_size=float(data_config.get('val_size', 0.1)),
        n_splits=data_config.get('n_splits', 5),
        split_method=data_config.get('split_method', 'kmer'),
        max_n_seqs=data_config.get('max_n_seqs', 10),
        vocab_size=data_config.get('vocab_size', 4),
        min_value=data_config.get('min_value', None),
        log_transform=data_config.get('log_transform', False),
    )
    return args

def load_train_config_from_yaml(config_path):
    """Load training + model sections from YAML into SimpleNamespace objects."""
    args = SimpleNamespace()

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    training_config = config.get('training', {})
    training_params = config.get('training_params', {})

    args.output_dir = training_config.get('output_dir', './results')
    args.random_state = training_config.get('random_state', 42)
    args.local_pairing = training_config.get('local_pairing', True)
    args.mode = training_config.get('mode', 'pairwise')
    args.resume = training_config.get('resume', True)
    args.resume_path = training_config.get('resume_path', None)
    args.use_multi_gpu = training_config.get('use_multi_gpu', False)
    args.skip_training = training_config.get('skip_training', False)

    args.train_batch_size = training_params.get('train_batch_size', 512)
    args.val_batch_size = training_params.get('val_batch_size', 512)
    args.test_batch_size = training_params.get('test_batch_size', 1024)
    args.num_workers = training_params.get('num_workers', 8)
    args.val_size = float(training_params.get('val_size', 0.1))
    args.max_epochs = training_params.get('max_epochs', 50)
    args.learning_rate = float(training_params.get('learning_rate', 0.001))
    args.weight_decay = float(training_params.get('weight_decay', 1e-4))
    args.patience = training_params.get('patience', 5)
    args.warmup_epochs = training_params.get('warmup_epochs', 5)
    args.scheduler_type = training_params.get('scheduler_type', 'plateau')
    args.grad_clip = float(training_params.get('grad_clip', 1.0))

    model_args = SimpleNamespace()
    model_config = config.get('model', None)
    if model_config is not None:
        model_args.models = model_config.get('models', None)
        model_args.hidden_dim = model_config.get('hidden_dim', 128)
        model_args.dropout = model_config.get('dropout', 0.1)

    return args, model_args

def load_generate_config_from_yaml(config_path):
    """Load generation config from YAML into a SimpleNamespace."""
    args = SimpleNamespace()

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    gc = config.get('generation', {})
    init_type = gc.get('init_type', 'freq_prior')
    args.init_type = init_type
    args.generate_dir = gc.get('generate_dir', str(GENERATION_DIR))
    args.model_name = gc.get('model_name', 'cnn')
    args.ckpt_dir = gc.get('ckpt_dir', None)
    args.num_sequences = gc.get('num_sequences', 100)
    args.top_k = gc.get('top_k', 100)
    args.reference_sequences = gc.get('reference_sequences', None)
    args.seq_length = gc.get('seq_length', None)
    args.skip_score_threshold = gc.get('skip_score_threshold', 0.3)
    args.skip_t_threshold = gc.get('skip_t_threshold', 0.05)
    args.score_temperature = gc.get('score_temperature', 20.0)

    gd = config.get('grad', {}) or {}
    args.num_iterations = gd.get('num_iterations', 50)
    args.refine_learning_rate = gd.get('refine_learning_rate', 0.05)
    args.temperature = gd.get('temperature', 1.0)
    args.alpha = gd.get('alpha', 0.1)
    args.beta = gd.get('beta', 1.5)
    args.improve_patience = gd.get('improve_patience', 20)
    args.freeze_patience = gd.get('freeze_patience', 15)
    args.generation_model_index = gd.get('generation_model_index')

    return args, config


def print_all_args(args):
    """Print all config arguments."""
    print("\n" + "="*80)
    print("Parameter configuration")
    print("="*80)

    for category, value in args.items():
        if isinstance(value, dict):
            print(f"\n[{category}]")
            for key in value:
                if isinstance(value[key], SimpleNamespace):
                    print(f"\n[{key}]")
                    for k, v in vars(value[key]).items():
                        print(f"  {k:25s} = {v}")
                else:
                    print(f"  {key:25s} = {value[key]}")
        else:
            print(f"{category:25s} = {value}")
    
    print("="*80 + "\n")

def merge_args(yaml_args, cli_args):
    """Merge YAML and CLI args; YAML values take precedence when not None."""
    merged = SimpleNamespace()
    for key, value in vars(cli_args).items():
        setattr(merged, key, value)
    for key, value in vars(yaml_args).items():
        if value is not None:
            setattr(merged, key, value)
    return merged

