"""
Dataset and collate utilities for DNA sequence training.

Sequence datasets:
    SequenceDataset        – pairwise (seq1_idx, seq2_idx, label)
    SingleSequenceDataset  – pointwise (seq_idx, score)

Embedding datasets (precomputed LLM embeddings):
    EmbeddingPairDataset   – pairwise (emb1, emb2, label)
    EmbeddingDataset       – pointwise (emb, score)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

from src.utils.config import BASE_TO_IDX, PAD_IDX, _BASES


def _seq_to_idx(seq):
    return torch.LongTensor([BASE_TO_IDX.get(b, PAD_IDX) for b in seq])


def _mutate(seq, rate):
    if rate <= 0:
        return seq
    s = list(seq)
    for i in range(len(s)):
        if np.random.random() < rate:
            s[i] = _BASES[np.random.randint(4)]
    return ''.join(s)


def _pad_seqs(seq_list):
    """Pad a list of 1-D LongTensors to equal length."""
    lengths = [len(s) for s in seq_list]
    if len(set(lengths)) == 1:
        return torch.stack(seq_list, dim=0)
    max_len = max(lengths)
    # print(f"  Padding sequences to max length {max_len}")
    out = torch.full((len(seq_list), max_len), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(seq_list):
        out[i, :lengths[i]] = s
    return out


def sequence_to_indices(sequence):
    """Convert a DNA sequence string to integer indices."""
    return _seq_to_idx(sequence)


# ===================== Collate functions =====================

def sequence_collate_fn(batch):
    """Collate for pairwise sequence data: (seq1, seq2, label)."""
    seq1_list, seq2_list, labels = zip(*batch)
    return _pad_seqs(seq1_list), _pad_seqs(seq2_list), torch.stack(labels)


def single_sequence_collate_fn(batch):
    """Collate for single-sequence data: (seq, label)."""
    seq_list, labels = zip(*batch)
    return _pad_seqs(seq_list), torch.stack(labels)


# ===================== Sequence datasets =====================

class SequenceDataset(Dataset):
    """Pairwise sequence dataset. Returns (seq1_indices, seq2_indices, label)."""

    def __init__(self, seq_pairs, labels, mutation_rate=0.0):
        self.seq_pairs = seq_pairs
        self.labels = (torch.FloatTensor(labels)
                       if not isinstance(labels, torch.Tensor) else labels)
        self.mutation_rate = mutation_rate

    def __len__(self):
        return len(self.seq_pairs)

    def __getitem__(self, idx):
        seq1, seq2 = self.seq_pairs[idx]
        seq1 = _mutate(seq1, self.mutation_rate)
        seq2 = _mutate(seq2, self.mutation_rate)
        return _seq_to_idx(seq1), _seq_to_idx(seq2), self.labels[idx]


class SingleSequenceDataset(Dataset):
    """Single-sequence dataset for pointwise regression."""

    def __init__(self, sequences, labels, use_log_transform=False):
        self.sequences = sequences
        labels_arr = np.array(labels, dtype=np.float32)
        if use_log_transform:
            self.labels = torch.FloatTensor(np.log(labels_arr + 1.0))
        else:
            self.labels = torch.FloatTensor(labels_arr)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return _seq_to_idx(self.sequences[idx]), self.labels[idx]


# ===================== Embedding datasets =====================

class EmbeddingPairDataset(Dataset):
    """Pair dataset backed by a ``{seq_str: tensor}`` embedding dict."""

    def __init__(self, pair_df, emb_dict):
        self.s1 = pair_df['s1'].tolist()
        self.s2 = pair_df['s2'].tolist()
        self.labels = torch.LongTensor(pair_df['label'].values.astype(int))
        self.emb = emb_dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.emb[self.s1[idx]], self.emb[self.s2[idx]], self.labels[idx]


class EmbeddingDataset(Dataset):
    """Single-sequence embedding dataset for pointwise regression."""

    def __init__(self, sequences, scores, emb_dict):
        self.seqs = sequences
        self.emb = emb_dict
        self.targets = torch.FloatTensor(np.array(scores, dtype=np.float32))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.emb[self.seqs[idx]], self.targets[idx]


# ============================= DataLoader factories ============================

def make_pairwise_loader(data, input_type, emb_dict=None, batch_size=512,
                         shuffle=True, mutation_rate=0.0, num_workers=0):
    """Build a pairwise DataLoader from a DataFrame with s1/s2/label columns."""
    if input_type == 'sequence':
        pairs = list(zip(data['s1'], data['s2']))
        ds = SequenceDataset(pairs, data['label'].values,
                             mutation_rate=mutation_rate)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, collate_fn=sequence_collate_fn)
    if input_type == 'embedding':
        ds = EmbeddingPairDataset(data, emb_dict)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    raise ValueError(
        f"Unknown input_type: {input_type} (supported: 'sequence', 'embedding')"
    )


def make_pointwise_loader(seqs, scores, input_type, emb_dict=None,
                          batch_size=1024, shuffle=True, num_workers=0):
    """Build a pointwise DataLoader from sequences and scores."""
    if input_type == 'sequence':
        ds = SingleSequenceDataset(seqs, scores, use_log_transform=False)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers,
                          collate_fn=single_sequence_collate_fn)
    if input_type == 'embedding':
        ds = EmbeddingDataset(seqs, scores, emb_dict)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    raise ValueError(
        f"Unknown input_type: {input_type} (supported: 'sequence', 'embedding')"
    )

