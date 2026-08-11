"""
Unified trainers for pairwise / pointwise cross-validation.

TrainEngine – single-model training / eval / pairwise-prob inference.
CVTrainer   – neural models (DL sequence or LLM embedding) cross-validation.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# ============================= Metrics ======================================

def compute_cls_metrics(targets, predictions, probs):
    """Classification metrics dict (accuracy, precision, recall, f1, auc, aupr)."""
    t = np.asarray(targets).ravel()
    p = np.asarray(predictions).ravel()
    pb = np.asarray(probs).ravel()
    has_both = len(np.unique(t)) > 1
    return {
        'accuracy': accuracy_score(t, p),
        'precision': precision_score(t, p, zero_division=0),
        'recall': recall_score(t, p, zero_division=0),
        'f1': f1_score(t, p, zero_division=0),
        'auc': roc_auc_score(t, pb) if has_both else 0.5,
        'aupr': average_precision_score(t, pb) if has_both else 0.5,
    }


def compute_reg_metrics(targets, predictions):
    """Regression metrics dict (spearman, ndcg@K, hit_rate@K)."""
    t = np.asarray(targets).ravel()
    p = np.asarray(predictions).ravel()
    res = {
        'spearman': spearmanr(t, p)[0],
    }
    
    n = len(t)
    if n >= 2:
        try:
            t_ranks = np.argsort(-t)
            p_ranks = np.argsort(-p)
            t_rel = t - np.min(t)
            
            for k in [20]:
                actual_k = min(k, n)
                
                # Value-based Hit Rate using Expected Hits for predicted ties
                t_threshold = t[t_ranks[actual_k - 1]]
                p_threshold = p[p_ranks[actual_k - 1]]
                
                # 1. Samples strictly above prediction threshold
                mask_above = p > p_threshold
                n_above = np.sum(mask_above)
                hits_above = np.sum(t[mask_above] >= t_threshold)
                
                # 2. Samples tied at the prediction threshold
                mask_tied = p == p_threshold
                n_tied = np.sum(mask_tied)
                hits_tied_total = np.sum(t[mask_tied] >= t_threshold)
                
                # 3. Expected hits from the tied group
                slots_remaining = actual_k - n_above
                if n_tied > 0:
                    expected_hits_tied = hits_tied_total * (slots_remaining / n_tied)
                else:
                    expected_hits_tied = 0.0
                
                expected_total_hits = hits_above + expected_hits_tied
                res[f'hit_rate_{k}'] = float(expected_total_hits / actual_k)
                
                try:
                    res[f'ndcg_{k}'] = ndcg_score([t_rel], [p], k=actual_k)
                except Exception:
                    res[f'ndcg_{k}'] = 0.0
        except Exception:
            pass

    return res


def winrate_reg_metrics(s1, s2, probs, seq_to_val):
    """Per-seq mean win-rate from pairwise probs → rank metrics vs true values.

    For each pair, the endpoint in ``seq_to_val`` is treated as the evaluated
    sequence (prefer s1 if both are). Win probability is P(test > opponent),
    then averaged per test sequence and correlated with true values.
    """
    if not seq_to_val:
        return None
    s1 = pd.Series(np.asarray(s1).astype(str).ravel())
    s2 = pd.Series(np.asarray(s2).astype(str).ravel())
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if len(s1) == 0 or len(probs) != len(s1):
        return None

    test_set = set(map(str, seq_to_val.keys()))
    s1_in = s1.isin(test_set)
    s2_in = s2.isin(test_set)
    test_seq = np.where(s1_in, s1, np.where(s2_in, s2, np.nan))
    valid = pd.notna(test_seq)
    if not valid.any():
        return None

    v_s1 = s1[valid].to_numpy()
    v_test = test_seq[valid]
    v_probs = probs[valid]
    win_probs = np.where(v_s1 == v_test, v_probs, 1.0 - v_probs)
    wr = (
        pd.DataFrame({"test_seq": v_test, "win_prob": win_probs})
        .groupby("test_seq", as_index=False)["win_prob"]
        .mean()
    )
    wr = wr[wr["test_seq"].isin(test_set)]
    if len(wr) < 2:
        return None
    true_vals = np.asarray([seq_to_val[s] for s in wr["test_seq"]], dtype=np.float64)
    return compute_reg_metrics(true_vals, wr["win_prob"].to_numpy())

