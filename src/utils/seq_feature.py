import numpy as np
from pathlib import Path
import sys
from typing import List
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.config import BASE_TO_IDX


def base_frequency(sequence):
    """计算基础核苷酸含量得分"""
    base_counts = {
        base: sequence.count(base) / len(sequence) for base in BASE_TO_IDX.keys()
    }

    return base_counts


def levenshtein_distance(s1, s2):
    """计算编辑距离（Levenshtein距离）"""
    import rapidfuzz
    return rapidfuzz.distance.Levenshtein.distance(s1, s2)


def kmer_set(seq: str, k: int = 3) -> set:
    """Set of distinct k-mers in *seq*."""
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def kmer_jaccard_distance(sequences: List[str], k: int = 3) -> np.ndarray:
    """Pairwise k-mer Jaccard distance matrix.

    Jaccard distance = 1 - |A & B| / |A | B|, computed on the **set**
    of distinct k-mers.  Position-free: shared motifs at different
    offsets are counted as overlap.

    Returns:
        np.ndarray [n, n] symmetric distance matrix in [0, 1].
    """
    n = len(sequences)
    kmer_sets = [kmer_set(s, k) for s in sequences]
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(kmer_sets[i] & kmer_sets[j])
            union = len(kmer_sets[i] | kmer_sets[j])
            d = 1.0 - inter / union if union > 0 else 1.0
            dist[i, j] = d
            dist[j, i] = d
    return dist
