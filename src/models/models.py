"""
DL encoder implementations for DNA sequence modelling.

All encoders inherit BaseEncoder and use fixed onehot encoding.
Supported architectures: cnn, lstm, transformer, cnn_lstm, cnn_transformer.

Factory functions:
    create_encoder(model_name, ...) -> BaseEncoder
    create_model(model_name, mode, ...) -> PairwiseModel | PointwiseModel
"""
import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from .base import BaseEncoder, EmbeddingEncoder, PairwiseModel, PointwiseModel
from src.data.dataloader import sequence_to_indices

# ---------------------------------------------------------------------------
# Default hyper-parameters per architecture
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    'cnn': {'num_filters': 256, 'kernel_sizes': [3, 5, 7]},
    'lstm': {'num_layers': 2},
    'transformer': {'num_heads': 8, 'num_layers': 6, 'dim_feedforward': 256},
    'hybrid': {'num_filters': 128, 'kernel_sizes': [3, 5, 7]},
}

# ============================= Encoders ====================================


class CNNEncoder(BaseEncoder):
    """Multi-scale Conv1d encoder."""

    def __init__(self, hidden_dim=128, num_filters=256,
                 kernel_sizes=(3, 5, 7), dropout=0.1, vocab_size=4):
        super().__init__(vocab_size)
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(self.vocab_size, hidden_dim)

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, num_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for k in kernel_sizes
        ])
        self.conv_proj = nn.Sequential(
            nn.Conv1d(num_filters * len(kernel_sizes), hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self._initialize_weights()

    def encode_sequence(self, seq_input, is_embedded: bool = False):
        x = seq_input if is_embedded else self._prepare_input(seq_input)
        x = x.transpose(1, 2)  # [B, H, L]
        x = self.conv_proj(torch.cat([cb(x) for cb in self.conv_blocks], dim=1))
        return self.pool(x).squeeze(-1)  # [B, H]


class LSTMEncoder(BaseEncoder):
    """Bidirectional LSTM encoder with last-hidden pooling."""

    def __init__(self, hidden_dim=128, num_layers=2, dropout=0.1, vocab_size=4):
        super().__init__(vocab_size)
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(self.vocab_size, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self._initialize_weights()

    def encode_sequence(self, seq_input, is_embedded: bool = False):
        x = seq_input if is_embedded else self._prepare_input(seq_input)
        _, (hidden, _) = self.lstm(x)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B, H]


# -------------------- RoPE Transformer building blocks ---------------------

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len):
        if seq_len > self.cos_cached.size(2):
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, max_seq_len=1024):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(S)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out)


class RoPETransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with RoPE and GELU."""

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1,
                 max_seq_len=1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiheadAttention(d_model, num_heads, dropout, max_seq_len)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(BaseEncoder):
    """RoPE Transformer encoder with mean pooling."""

    def __init__(self, hidden_dim=128, num_heads=8, num_layers=6,
                 dim_feedforward=256, dropout=0.1, vocab_size=4):
        super().__init__(vocab_size)
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(self.vocab_size, hidden_dim)
        self.embed_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=hidden_dim, num_heads=num_heads,
                dim_feedforward=dim_feedforward, dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self._initialize_weights()

    def encode_sequence(self, seq_input, is_embedded: bool = False):
        x = seq_input if is_embedded else self._prepare_input(seq_input)
        x = self.embed_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x.mean(dim=1)  # [B, H]


class HybridEncoder(BaseEncoder):
    """Optional CNN trunk + global context module (lstm or transformer).

    Supports: cnn_lstm, cnn_transformer (use_cnn=True),
              and global-context-only ablation (use_cnn=False).
    """

    def __init__(self, global_module='transformer', hidden_dim=128,
                 num_filters=128, kernel_sizes=(3, 5, 7),
                 dropout=0.3, vocab_size=4):
        super().__init__(vocab_size)
        self.global_module_name = global_module
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(self.vocab_size, hidden_dim)

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, num_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for k in kernel_sizes
        ])
        self.conv_proj = nn.Sequential(
            nn.Conv1d(num_filters * len(kernel_sizes), hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        if global_module == 'lstm':
            self.seq_module = nn.LSTM(
                input_size=hidden_dim, hidden_size=hidden_dim // 2,
                num_layers=1, batch_first=True, bidirectional=True,
                dropout=dropout,
            )

        elif global_module == 'transformer':
            self.seq_module = RoPETransformerEncoderLayer(
                d_model=hidden_dim, num_heads=4,
                dim_feedforward=hidden_dim * 2, dropout=dropout,
            )

        else:
            raise ValueError(f"Unknown global module: {global_module}")

        self.seq_norm = nn.LayerNorm(hidden_dim)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._initialize_weights()

    def encode_sequence(self, seq_input, is_embedded: bool = False):
        x = seq_input if is_embedded else self._prepare_input(seq_input)

        x_t = x.transpose(1, 2)  # [B, H, L]
        x = self.conv_proj(
            torch.cat([cb(x_t) for cb in self.conv_blocks], dim=1))
        x = x.transpose(1, 2)  # [B, L, H]

        residual = x
        if self.global_module_name == 'lstm':
            out, _ = self.seq_module(x)
        elif self.global_module_name == 'transformer':
            out = self.seq_module(x)
        x = self.seq_norm(residual + out)

        pooled = x.max(dim=1)[0] + x.mean(dim=1)
        return self.proj(pooled)  # [B, H]


# ============================= Factories ===================================

def create_model(model_name, input_type='sequence', mode='pairwise',
                 hidden_dim=128, dropout=0.1, emb_dim=0, vocab_size=4, **kwargs):
    """Create a full model (encoder + head) for the given training mode.

    Args:
        model_name: encoder architecture name (ignored when input_type='embedding').
        mode: 'pairwise' or 'pointwise'.
        hidden_dim: encoder hidden dimension.
        dropout: dropout rate.
        emb_dim: embedding dim when input_type='embedding'.
        input_type: 'sequence' or 'embedding' (hybrid is not supported).
    """
    if input_type == 'embedding':
        encoder = EmbeddingEncoder(emb_dim, hidden_dim)
    else:
        encoder = create_encoder(model_name, hidden_dim=hidden_dim,
                                 dropout=dropout, vocab_size=vocab_size, **kwargs)

    if mode == 'pointwise':
        return PointwiseModel(encoder, hidden_dim=hidden_dim, dropout=dropout)
    else:
        return PairwiseModel(encoder, hidden_dim=hidden_dim, dropout=dropout)


def create_encoder(model_name, hidden_dim=128, dropout=0.1, vocab_size=4, **kwargs):
    """Instantiate an encoder by name.

    Supported names: cnn, lstm, transformer,
                     cnn_lstm, cnn_transformer.
    """
    if model_name == 'cnn':
        cfg = MODEL_CONFIG['cnn']
        return CNNEncoder(
            hidden_dim=hidden_dim,
            num_filters=kwargs.get('num_filters', cfg['num_filters']),
            kernel_sizes=kwargs.get('kernel_sizes', cfg['kernel_sizes']),
            dropout=dropout,
            vocab_size=vocab_size,
        )
    elif model_name == 'lstm':
        cfg = MODEL_CONFIG['lstm']
        return LSTMEncoder(
            hidden_dim=hidden_dim,
            num_layers=kwargs.get('num_layers', cfg['num_layers']),
            dropout=dropout,
            vocab_size=vocab_size,
        )
    elif model_name == 'transformer':
        cfg = MODEL_CONFIG['transformer']
        return TransformerEncoder(
            hidden_dim=hidden_dim,
            num_heads=kwargs.get('num_heads', cfg['num_heads']),
            num_layers=kwargs.get('num_layers', cfg['num_layers']),
            dim_feedforward=kwargs.get('dim_feedforward', cfg['dim_feedforward']),
            dropout=dropout,
            vocab_size=vocab_size,
        )
    elif '_' in model_name:
        assert model_name.split('_')[0] == 'cnn', f"Unknown model name: {model_name}"
        cfg = MODEL_CONFIG['hybrid']
        return HybridEncoder(
            global_module=model_name.split('_')[1],
            hidden_dim=hidden_dim,
            num_filters=kwargs.get('num_filters', cfg['num_filters']),
            kernel_sizes=kwargs.get('kernel_sizes', cfg['kernel_sizes']),
            dropout=dropout,
            vocab_size=vocab_size,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# ============================= EnsembleModel ================================

class EnsembleModel(nn.Module):
    """Ensemble of K-fold models for inference and generation.

    Loads fold checkpoints from:  ckpt_dir/{model_name}_fold*_lr{best_lr}.pt
    """

    def __init__(
        self,
        model_name: List[str],
        ckpt_dir: str = None,
        device: str = 'cuda',
        hidden_dim: int = 128,
        dropout: float = 0.1,
        vocab_size: int = 4,
        score_temperature: int = 1.0,
        **encoder_kwargs,
    ):

        super().__init__()

        if isinstance(model_name, str):
            model_name = [model_name]

        # read all the checkpoints in the ckpt_dir
        ckpt_files = []
        for f in os.listdir(ckpt_dir):
            name = f.split('_fold')[0]
            if name in model_name:
                ckpt_files.append([name, f"{ckpt_dir}/{f}"])
        ckpt_files.sort(key=lambda x: (x[0], x[1]))

        if len(ckpt_files) == 0:
            raise FileNotFoundError(
                f"No checkpoints found for {model_name} in {ckpt_dir}")

        models = []
        self.model_names: List[str] = []
        for name, ckpt_file in ckpt_files:
            model = create_model(
                name,
                mode='pairwise',
                hidden_dim=hidden_dim,
                dropout=dropout,
                vocab_size=vocab_size,
                **encoder_kwargs
            )

            state = torch.load(str(ckpt_file), map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            models.append(model)
            self.model_names.append(name)
            print(f"Loaded {ckpt_file}")

        # Register so .eval()/.train() and .to() apply to every fold model.
        self.models = nn.ModuleList(models)

        # Preserve requested order of model types for reporting.
        seen = set()
        self.model_types: List[str] = []
        for n in model_name:
            if n in self.model_names and n not in seen:
                self.model_types.append(n)
                seen.add(n)
        for n in self.model_names:
            if n not in seen:
                self.model_types.append(n)
                seen.add(n)

        print(
            f"EnsembleModel initialized: {len(self.models)} folds "
            f"across {self.model_types}"
        )

        self.score_temperature = score_temperature
        self.hidden_dim = hidden_dim
        self.device = device
        self.eval()

    def train(self, mode: bool = True):
        """Keep fold models in sync with the ensemble train/eval flag."""
        super().train(mode)
        for m in self.models:
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # ── helpers ────────────────────────────────────────────────────────────

    def seq_to_indices(self, seq: str) -> torch.Tensor:
        """Convert DNA string to index tensor ``[1, L]``."""
        return sequence_to_indices(seq).unsqueeze(0).to(self.device)

    def _type_indices(self, model_type: str) -> List[int]:
        return [i for i, n in enumerate(self.model_names) if n == model_type]

    @staticmethod
    def _mean_std(values: np.ndarray) -> Tuple[float, float]:
        values = np.asarray(values, dtype=np.float64)
        mean = float(values.mean()) if values.size else float('nan')
        std = float(values.std()) if values.size > 1 else 0.0
        return mean, std

    # ── pairwise scoring ───────────────────────────────────────────────────

    def _score(self, model, seq_enc, ref_enc, temperature):
        """P(seq > ref) for rank-loss: sigmoid((s_seq - s_ref) / T)."""
        logit = (model.head(seq_enc) - model.head(ref_enc)).squeeze(-1)
        return torch.sigmoid(logit / temperature)

    # ── ensemble prediction ────────────────────────────────────────────────

    def calculate_prediction_scores(
        self, seq: str, ref_indices: torch.Tensor,
        per_model: bool = False,
    ) -> torch.Tensor:
        """Ensemble P(seq > ref).

        Args:
            seq: DNA sequence string.
            ref_indices: ``[N, L]`` integer-encoded reference sequences.
            per_model: if True return ``[K, N]`` per-fold scores,
                       otherwise return ``[N]`` averaged across folds.
        """
        was_training = self.training
        self.eval()
        seq_idx = self.seq_to_indices(seq)
        n_refs = ref_indices.shape[0]

        all_scores = []
        with torch.no_grad():
            for model in self.models:
                seq_enc = model.encoder.encode_sequence(seq_idx)
                ref_enc = model.encoder.encode_sequence(ref_indices)
                seq_exp = seq_enc.expand(n_refs, -1)

                sc = self._score(
                    model, seq_exp, ref_enc, self.score_temperature)

                all_scores.append(sc)

        if was_training:
            self.train()

        stacked = torch.stack(all_scores)  # [K, N]
        return stacked if per_model else stacked.mean(dim=0)

    def score_sequence(
        self, seq: str, ref_indices: torch.Tensor,
    ) -> Dict[str, object]:
        """Score one sequence; return ensemble + per-model-type stats.

        Fitness for each fold is mean P(seq > ref) over ``ref_indices``.
        Uncertainty is the std of fold fitnesses within the group.
        """
        per_model = self.calculate_prediction_scores(
            seq, ref_indices, per_model=True)  # [K, N]
        fold_means = per_model.mean(dim=1).detach().cpu().numpy()

        ens_score, ens_unc = self._mean_std(fold_means)
        by_type: Dict[str, Dict[str, float]] = {}
        for mtype in self.model_types:
            idx = self._type_indices(mtype)
            score, unc = self._mean_std(fold_means[idx])
            by_type[mtype] = {'score': score, 'uncertainty': unc}

        return {
            'score': ens_score,
            'uncertainty': ens_unc,
            'fold_scores': fold_means.tolist(),
            'by_type': by_type,
        }

    def score_sequences(
        self, sequences: List[str], ref_indices: torch.Tensor,
    ) -> List[Dict[str, object]]:
        """Score many sequences with the same metric as ``score_sequence``."""
        return [self.score_sequence(seq, ref_indices) for seq in sequences]

    def evaluate_sequences_batch(
        self, sequences: List[str], ref_indices: torch.Tensor,
    ) -> List[float]:
        """Score multiple sequences against a reference pool.

        Returns one float (mean over refs / folds) per sequence.
        """
        return [
            self.score_sequence(seq, ref_indices)['score']
            for seq in sequences
        ]

    # ── encoding (for embeddings / visualization) ──────────────────────────

    def encode_sequence(self, seq_input):
        """Average encoding across all fold models.

        Args:
            seq_input: ``[B, L]`` indices.

        Returns:
            ``[B, H]`` averaged encoding.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            encodings = [
                m.encoder.encode_sequence(seq_input)
                for m in self.models
            ]
            out = torch.stack(encodings).mean(dim=0)
        if was_training:
            self.train()
        return out
