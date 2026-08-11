from pathlib import Path

# if more bases are added, update the BASE_TO_IDX
BASE_TO_IDX = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
IDX_TO_BASE = {v: k for k, v in BASE_TO_IDX.items()}
VOCAB_SIZE = len(BASE_TO_IDX)
PAD_IDX = len(BASE_TO_IDX)
_BASES = list(BASE_TO_IDX.keys())

# Default experiment directories
# MODEL_RESULT_DIR = Path("./experiments/model")
MODEL_RESULT_DIR = Path("/root/autodl-tmp/NADesign/PROBE-all/experiments/evaluation")
GENERATION_DIR = Path("./experiments/generation")
EVALUATION_DIR = Path("./experiments/evaluation")
VISUALIZATION_DIR = Path("./experiments/visualization")

DATASET_ORDER = ['CDCP1', 'ITGA3', 'LAR', 'NRP1', 'NRP2', 'PTK7']

DATA_NAMES = DATASET_ORDER + ['PDL1']

DATASET_DISPLAY_NAMES = {
    'CDCP1': 'CDCP1',
    'ITGA3': 'ITGA3',
    'LAR': 'PTPRD/F/S',
    'NRP1': 'NRP1',
    'NRP2': 'NRP2',
    'PTK7': 'PTK7',
    'PDL1': 'PD-L1',
}

# Regression metrics used in evaluation / visualization
REGRESSION_METRICS = [
    "spearman",
    "hit_rate_20",
    "ndcg_20",
]

# Model order
LLM_MODEL_CONFIG = ['evo2', 'alphagenome']
DL_MODEL_CONFIG = ['cnn', 'lstm', 'transformer']
ADDITIONAL_MODEL_CONFIG = ['cnn_lstm', 'cnn_transformer']

MODEL_CONFIG = DL_MODEL_CONFIG + ADDITIONAL_MODEL_CONFIG + LLM_MODEL_CONFIG
SIMPLE_MODEL_CONFIG = DL_MODEL_CONFIG + LLM_MODEL_CONFIG
# Model display names
MODEL_DISPLAY_NAMES = {
    'evo2': 'Evo2',
    'alphagenome': 'AlphaGenome',
    'cnn': 'CNN',
    'lstm': 'LSTM',
    'transformer': 'Transformer',
    'cnn_transformer': 'CNN-Transformer',
    'cnn_lstm': 'CNN-LSTM',
}

MODE_PALETTE = {
    'pairwise': '#DC0000',
    'pointwise': '#A2AFD2', 
}

MODEL_PALETTE = {
    # 'evo2': '#DC0000', 
    # 'alphagenome': '#00A087',
    'evo2': '#00A087', # NPG Deep Blue 
    'alphagenome': '#3C5488', # NPG Teal 
    'transformer': '#F39B7F',   # NPG Orange 
    'lstm': '#7030A0',  # NPG Purple
    'cnn': '#F1C40F', # NPG Yellow
    'cnn_transformer': '#E64B35',
    'cnn_lstm': '#4DBBD5',
}


def compute_sig_mark(pval: float) -> str:
    """Compute significance marker from p-value."""
    if pval < 0.0001:
        return '****'
    if pval < 0.001:
        return '***'
    if pval < 0.01:
        return '**'
    if pval < 0.05:
        return '*'
    return 'n.s.'
