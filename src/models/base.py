"""
Base model classes for pairwise and pointwise training modes.

BaseEncoder      – abstract encoder with fixed onehot DNA encoding (vocab_size-dim).
EmbeddingEncoder – encoder for precomputed embeddings (e.g. LLM).
PairwiseModel    – encoder wrapper for pairwise rank training.
PointwiseModel   – encoder wrapper for pointwise regression (MSE loss).
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import VOCAB_SIZE

def load_embeddings(emb_path, model_name):
    # Support both Evo2 NIM and AlphaGenome embeddings
    if model_name == 'alphagenome':
        emb_path = f'{emb_path}/alphagenome_embeddings.pt'
    elif model_name == 'evo2':
        emb_path = f'{emb_path}/evo2_embeddings_b20.pt'
        
    if not Path(emb_path).exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    emb_raw = torch.load(emb_path, map_location='cpu')
    emb_dict = {k: v.float() for k, v in emb_raw.items()}
    emb_dim = next(iter(emb_dict.values())).shape[0]
    print(f"  Loaded embeddings: n={len(emb_dict)}, dim={emb_dim}")

    all_embs = torch.stack(list(emb_dict.values()))
    mean_e = all_embs.mean(dim=0)
    std_e = all_embs.std(dim=0) + 1e-8
    for k in emb_dict:
        emb_dict[k] = (emb_dict[k] - mean_e) / std_e
    
    return emb_dict, emb_dim

class BaseEncoder(nn.Module):
    """Abstract encoder with fixed onehot DNA encoding.

    Subclasses must override ``encode_sequence`` to map
    sequence tensors to ``[B, hidden_dim]`` feature vectors.

    Notes:
        - Dataloaders provide integer indices ``[B, L]``
          where A=0, T=1, G=2, C=3, ..., unknown/pad = vocab_size.
        - Encoders convert indices to one-hot internally:
          A=(1,0,0,0,...,0), T=(0,1,0,0,...,0), G=(0,0,1,0,...,0), C=(0,0,0,1,...,0),
          ..., unknown -> (0,0,0,0,...,0).
        - You may also pass one-hot directly as ``[B, L, vocab_size]``.
    """ 

    def __init__(self, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.vocab_size = vocab_size

    def onehot_encode(self, seq_indices: torch.Tensor) -> torch.Tensor:
        """Convert integer-encoded DNA to onehot ``[B, L, vocab_size]``."""
        if seq_indices.dim() == 3 and seq_indices.size(-1) == self.vocab_size:
            return seq_indices.float()
        # PAD/unknown (== vocab_size) must not enter one_hot; zero those positions.
        valid_mask = (seq_indices >= 0) & (seq_indices < self.vocab_size)
        clamped = seq_indices.clamp(0, self.vocab_size - 1)
        oh = F.one_hot(clamped, num_classes=self.vocab_size).float()
        return oh * valid_mask.unsqueeze(-1).float()

    def _prepare_input(self, seq_input: torch.Tensor) -> torch.Tensor:
        """Prepare input for encoding.

        *seq_input* is ``[B, L]`` integer indices that get one-hot encoded then projected.
        """
        return self.input_proj(self.onehot_encode(seq_input))

    def encode_sequence(self, seq_input: torch.Tensor,
                        is_embedded: bool = False) -> torch.Tensor:
        """Encode ``[B, L]`` integer indices (or ``[B, L, H]`` if
        *is_embedded*) to ``[B, hidden_dim]``."""
        raise NotImplementedError

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class EmbeddingEncoder(nn.Module):
    """Encoder for precomputed embeddings (e.g. LLM).

    Projects ``emb_dim`` → ``hidden_dim`` with LayerNorm + GELU.
    Used as a drop-in replacement for BaseEncoder when the input
    is a fixed-size embedding vector rather than a DNA sequence.
    """

    def __init__(self, emb_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def encode_sequence(self, emb):
        return self.proj(emb)


def _make_head(input_dim, output_dim, dropout=0.1):
    """Build a small MLP classification / regression head."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(128, output_dim),
    )


class PairwiseModel(nn.Module):
    """Wraps an encoder for pairwise rank training (MarginRankingLoss)."""

    def __init__(self, encoder, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.loss_type = 'rank'
        self.hidden_dim = hidden_dim
        self.head = _make_head(hidden_dim, 1, dropout)

    def _encode(self, seq):
        return self.encoder.encode_sequence(seq)

    def forward(self, seq1, seq2=None, return_prob=False):
        e1 = self._encode(seq1)
        score1 = self.head(e1)
        if seq2 is not None:
            e2 = self._encode(seq2)
            score2 = self.head(e2)
            if return_prob:
                return torch.sigmoid(score1 - score2)  # [B, 1]
            else:
                return torch.cat([score1, score2], dim=1)  # [B, 2]
        else:
            return score1


class PointwiseModel(nn.Module):
    """Wraps an encoder for pointwise (regression) training.

    forward(seq) -> scalar score, trained with MSELoss.
    """

    def __init__(self, encoder, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.head = _make_head(hidden_dim, 1, dropout)

    def _encode(self, seq):
        return self.encoder.encode_sequence(seq)

    def forward(self, seq1, seq2=None, return_prob=False):
        e1 = self._encode(seq1)
        score1 = self.head(e1)
        if seq2 is not None:
            e2 = self._encode(seq2)
            score2 = self.head(e2)
            if return_prob:
                return torch.sigmoid(score1 - score2)  # [B, 1]
            else:
                return torch.cat([score1, score2], dim=1)  # [B, 2]
        else:
            return score1
