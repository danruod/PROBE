#!/usr/bin/env python3
"""
PROBE sequence optimization pipeline:
    Initialization → Gradient Refinement

"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from time import time
import sys
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.dataloader import sequence_to_indices, _pad_seqs
from src.utils.config import (
    BASE_TO_IDX, IDX_TO_BASE, VOCAB_SIZE, _BASES,
)
from src.models.models import EnsembleModel
from src.utils.seq_feature import base_frequency, kmer_jaccard_distance
from src.utils.plot import plot_score_history

class PROBEGenerator:
    """PROBE sequence generator.

    Uses an ensemble fitness model to evaluate and optimize DNA sequences
    against a reference pool of known high-affinity aptamers.

    Pipeline stages:
        1. Initialization  – diverse seed population (topk / random / reference)
        2. Gradient Refine  – Gumbel-Softmax optimization per candidate
    """

    def __init__(
        self,
        model: EnsembleModel,
        reference_pool: List[str],
        seq_length: int = None,
        skip_t_threshold: float = 0.0,
        top_k: int = 100,
        seed: int = 2026,
        generate_dir: str = 'experiments/generate',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model = model
        self.device = device

        self.reference_pool = reference_pool or []
        self.ref_pool_indices = None
        self._ref_array: Optional[np.ndarray] = None

        if self.reference_pool:
            self._prepare_reference_pool()

        if seq_length is not None:
            self.seq_length = seq_length
        elif self.ref_pool_indices is not None:
            self.seq_length = self.ref_pool_indices.shape[1]
        else:
            raise ValueError("seq_length must be specified when reference_pool is empty")

        self.skip_t_threshold = skip_t_threshold
        
        self.top_k = min(top_k, len(reference_pool))

        self.generate_dir = generate_dir

        # Unified per-step score history
        # Each entry: {step, stage, best, mean, inter_div, min_ref_dist}
        self.score_history: List[Dict] = []

    def _prepare_reference_pool(self):
        """Convert reference pool strings → ``[N, L]`` index tensor."""
        idx_list = [sequence_to_indices(seq) for seq in self.reference_pool]
        self.ref_pool_indices = _pad_seqs(idx_list).to(self.device)
        self._ref_array = np.array([list(s) for s in self.reference_pool])

    # ── diversity metrics ─────────────────────────────────────────────────

    @staticmethod
    def _kmer_profile(seq: str, k: int = 3) -> np.ndarray:
        """4^k-dim frequency vector for k-mers in *seq*."""
        size = 4 ** k
        base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        counts = np.zeros(size, dtype=np.float32)
        for i in range(len(seq) - k + 1):
            idx = 0
            for j in range(k):
                b = base_map.get(seq[i + j])
                if b is None:
                    break
                idx = idx * 4 + b
            else:
                counts[idx] += 1
        total = counts.sum()
        return counts / total if total > 0 else counts

    def _compute_diversity(self, sequences: List[str]) -> Dict[str, float]:
        """Compute diversity + homology/family metrics.

        Returns:
            n_families:     number of sequence families (k-mer Jaccard, threshold 0.5)
            family_entropy: Shannon entropy of cluster-size distribution
            kmer_div_ref:   mean Jensen-Shannon divergence of 3-mer profiles
                            between generated seqs and reference pool
        """
        arr = np.array([list(s) for s in sequences])  # [B, L]
        L = arr.shape[1]
        n = len(sequences)

        # ── family clustering (single-linkage on k-mer Jaccard distance) ──
        jaccard_threshold = 0.5
        if n > 1:
            jd = kmer_jaccard_distance(sequences, k=3)
            labels = list(range(n))
            for i in range(n):
                for j in range(i + 1, n):
                    if jd[i, j] < jaccard_threshold:
                        ri, rj = labels[i], labels[j]
                        if ri != rj:
                            for q in range(n):
                                if labels[q] == rj:
                                    labels[q] = ri
            clusters = np.array(labels)
            unique, counts = np.unique(clusters, return_counts=True)
            n_families = len(unique)
            p = counts / counts.sum()
            family_entropy = float(-np.sum(p * np.log2(p + 1e-12)))
        else:
            n_families = 1
            family_entropy = 0.0

        # ── k-mer divergence to reference pool ──
        kmer_div_ref = 0.0
        if self._ref_array is not None and len(self._ref_array) > 0:
            ref_seqs = [''.join(row) for row in self._ref_array[:, :L]]
            ref_profiles = np.array([self._kmer_profile(s) for s in ref_seqs])
            ref_mean = ref_profiles.mean(axis=0)
            gen_profiles = np.array([self._kmer_profile(s) for s in sequences])
            divs = np.empty(n)
            for i in range(n):
                m = 0.5 * (gen_profiles[i] + ref_mean)
                m = np.clip(m, 1e-12, None)
                kl_p = np.sum(gen_profiles[i] * np.log(gen_profiles[i] / m + 1e-12))
                kl_q = np.sum(ref_mean * np.log(ref_mean / m + 1e-12))
                divs[i] = 0.5 * (kl_p + kl_q)
            kmer_div_ref = float(divs.mean())

        return {
            'n_families': n_families,
            'family_entropy': family_entropy,
            'kmer_div_ref': kmer_div_ref,
        }

    # ======================= evaluate sequences ==============================

    def _per_model_fitness(
        self, seq: str, ref_indices: torch.Tensor,
    ) -> np.ndarray:
        """Per-fold mean sigmoid fitness for one discrete sequence. Shape [K]."""
        per_model = self.model.calculate_prediction_scores(
            seq, ref_indices, per_model=True)  # [K, N]
        return per_model.mean(dim=1).cpu().numpy()

    def _rank_fitness(
        self,
        model,
        seq_enc: torch.Tensor,
        ref_enc: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Mean P(seq > ref) over the reference pool. Returns ``[B]``."""
        B, H = seq_enc.shape
        N = ref_enc.shape[0]
        seq_flat = seq_enc.unsqueeze(1).expand(-1, N, -1).reshape(-1, H)
        ref_flat = ref_enc.unsqueeze(0).expand(B, -1, -1).reshape(-1, H)
        logit = (model.head(seq_flat) - model.head(ref_flat)).view(B, N)
        return torch.sigmoid(logit / temperature).mean(dim=1)

    @staticmethod
    def _decode_logits(batch_logits: torch.Tensor) -> List[str]:
        """Argmax decode ``[B, L, V]`` logits to DNA strings."""
        idx = torch.argmax(batch_logits, dim=-1).detach().cpu().numpy()
        return [''.join(IDX_TO_BASE[int(j)] for j in row) for row in idx]

    @staticmethod
    def _confidence_level(model_scores: np.ndarray) -> Dict:
        mu = float(model_scores.mean())
        sigma = float(model_scores.std())
        if mu > 0.85 and sigma < 0.05:
            level = 'high'
        elif mu > 0.85 and sigma > 0.15:
            level = 'low'
        else:
            level = 'medium'
        return {
            'model_scores': model_scores.tolist(),
            'mu': mu,
            'sigma': sigma,
            'confidence': level,
        }

    def evaluate_sequences(self, sequences: List[str],
                            top_k: int = None) -> List[float]:
        """Score sequences against the top-k reference pool.

        Uses the same eval-mode metric as ``scripts/score.py`` /
        ``EnsembleModel.score_sequence`` (mean P(seq > ref) over refs/folds).
        """
        if self.ref_pool_indices is None:
            raise ValueError("reference_pool is required for evaluation")

        k = self.top_k if top_k is None else top_k
        pool = self.ref_pool_indices[:min(k, len(self.ref_pool_indices))]
        return self.model.evaluate_sequences_batch(
            sequences=sequences,
            ref_indices=pool,
        )

    # ======================= Initialization ==============================

    def _t_fraction(self, seq: str) -> float:
        """Fraction of T bases in *seq*."""
        return seq.count('T') / max(len(seq), 1)

    def _meets_t_threshold(self, seq: str) -> bool:
        """True if T fraction is at least ``skip_t_threshold`` (or threshold is off)."""
        if self.skip_t_threshold <= 0:
            return True
        return self._t_fraction(seq) >= self.skip_t_threshold

    def _random_sequence(self, length, p=None):
        if p is None:
            p = [0.25, 0.25, 0.25, 0.25]
        return ''.join(np.random.choice(_BASES, size=length, p=p))

    def _random_sequence_with_t(self, length, p=None, max_tries: int = 100) -> str:
        """Sample a random sequence that meets ``skip_t_threshold`` when set."""
        for _ in range(max_tries):
            seq = self._random_sequence(length, p=p)
            if self._meets_t_threshold(seq):
                return seq
        # Fallback: force-insert the minimum number of T's.
        seq = list(self._random_sequence(length, p=p))
        min_t = max(1, int(np.ceil(self.skip_t_threshold * length)))
        positions = np.random.choice(length, size=min(min_t, length), replace=False)
        for pos in positions:
            seq[pos] = 'T'
        return ''.join(seq)

    def initialize_sequences(self, init_type: str = 'freq_prior', num_sequences: int = 100) -> List[str]:
        """Build a diverse seed population.

        Returns: sequences
        """
        sequences = []

        if init_type == 'freq_prior':
            freqs = np.mean(
                [list(base_frequency(s).values()) for s in self.reference_pool],
                axis=0)
            freqs = np.asarray(freqs, dtype=float)
            freqs = np.clip(freqs, 0.0, None)
            total = float(freqs.sum())
            # Sequences may contain N/other bases → A/T/G/C masses need renormalization.
            freqs = (
                np.full(4, 0.25, dtype=float) if total <= 1e-12
                else freqs / total
            )
            for _ in range(num_sequences):
                sequences.append(
                    self._random_sequence_with_t(self.seq_length, p=freqs))
        elif init_type == 'random':
            for _ in range(num_sequences):
                sequences.append(self._random_sequence_with_t(self.seq_length))
        elif init_type == 'topk':
            # Prefer reference seqs that already meet the T threshold.
            t_ok = [s for s in self.reference_pool if self._meets_t_threshold(s)]
            pool = t_ok if t_ok else self.reference_pool
            for _ in range(num_sequences):
                sequences.append(pool[np.random.randint(0, len(pool))])
        else:
            raise ValueError(
                f"Invalid init type: {init_type}. "
                f"Expected one of: freq_prior, random, topk"
            )

        n_ok = sum(1 for s in sequences if self._meets_t_threshold(s))
        if self.skip_t_threshold > 0:
            print(f"  Init T≥{self.skip_t_threshold:.2f}: "
                  f"{n_ok}/{len(sequences)} sequences")

        return sequences

    def filter_candidates(
        self,
        candidates: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Drop candidates below score / T-content thresholds."""
        filtered = []
        for seq, score in candidates:
            if not self._meets_t_threshold(seq):
                continue
            filtered.append((seq, score))
        return filtered

    # ======================= Gradient Refinement =========================

    def gradient_refine(
        self,
        seqs: List[str],
        num_iterations: int = 100,
        learning_rate: float = 0.1,
        temperature: float = 1.0,
        alpha: float = 0.1,
        beta: float = 1.5,
        improve_patience: int = 20,
        freeze_patience: int = 15,
        generation_model_index: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Gradient ascent with ensemble LCB (or single-model) objective.

        Objective (default): maximize ``μ - beta * σ`` over ensemble members.
        If ``generation_model_index`` is set, only that member provides the
        optimization gradient; all members still evaluate mean / uncertainty.

        Diversity term (``alpha > 0``): penalize mean pairwise cosine
        similarity of batch embeddings to reduce mode collapse.

        T-content constraint (``skip_t_threshold > 0``): soft hinge penalty
        when soft T fraction falls below the threshold, and hard gate when
        accepting historical-best discrete sequences.

        Stopping / freezing:
          - ``improve_patience``: stop when population mean of historical-best
            fitness does not improve for this many consecutive iters (0=off).
          - ``freeze_patience``: freeze logits of a sequence that has not
            improved its own best for this many iters (0=off). Frozen seqs
            still act as fixed diversity anchors when ``alpha > 0``.

        Returns:
            [(best_seq, best_score), ...]
        """
        t0 = time()
        if self.model is None:
            raise ValueError("Model is not initialized")
        if self.ref_pool_indices is None:
            raise ValueError("reference_pool is required for gradient refine")
        if not seqs:
            return []

        B = len(seqs)
        L = len(seqs[0])
        if any(len(s) != L for s in seqs):
            raise ValueError("All sequences must have the same length")

        ref_indices = self.ref_pool_indices[:self.top_k]
        K = len(self.model.models)
        T = float(self.model.score_temperature)
        use_diversity = alpha > 0 and B > 1
        use_t_constraint = self.skip_t_threshold > 0
        # Soft hinge weight: strong enough to resist fitness-driven T collapse.
        t_loss_weight = 10.0
        t_base_idx = BASE_TO_IDX['T']
        use_improve_stop = improve_patience > 0
        use_freeze = freeze_patience > 0
        single_model = generation_model_index is not None
        improve_eps = 1e-5

        if single_model and not 0 <= generation_model_index < K:
            raise ValueError(
                f"generation_model_index must be in [0, {K - 1}], "
                f"got {generation_model_index}")

        # ── initial evaluation ──
        self.model.eval()
        self.score_history = []
        init_model_scores = np.zeros((B, K), dtype=np.float64)
        for i, seq in enumerate(seqs):
            init_model_scores[i] = self._per_model_fitness(seq, ref_indices)
        initial_scores = init_model_scores.mean(axis=1)

        self.score_history.append({
            'step': 0,
            'best': float(initial_scores.max()),
            'mean': float(initial_scores.mean()),
            'scores': initial_scores.tolist(),
            'sigma': init_model_scores.std(axis=1).tolist(),
            'n_active': B,
            **self._compute_diversity(seqs),
        })

        print(f"  Scoring against {ref_indices.shape[0]} reference sequences")
        print(f"  Initial scores: best={initial_scores.max():.4f}, "
              f"mean={initial_scores.mean():.4f}, "
              f"σ={init_model_scores.std(axis=1).mean():.4f}")

        best_scores = initial_scores.copy()
        best_seqs = list(seqs)
        best_model_scores = init_model_scores.copy()
        best_select = (
            init_model_scores[:, generation_model_index].copy()
            if single_model else initial_scores.copy())

        # ── logits & optimizer ──
        init_scale = 5.0
        logits_list = []
        for seq in seqs:
            idx = sequence_to_indices(seq).to(self.device)
            oh = F.one_hot(idx, num_classes=VOCAB_SIZE).float()
            logits_list.append(oh * init_scale + torch.randn_like(oh) * 0.01)
        batch_logits = torch.stack(logits_list).requires_grad_(True)

        t0_sched = max(10, num_iterations // 4)
        t_mult = 2
        optimizer = torch.optim.AdamW(
            [batch_logits], lr=learning_rate, betas=(0.9, 0.999),
            weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0_sched, T_mult=t_mult, eta_min=learning_rate * 0.01)

        restart_iters = set()
        period, pos = t0_sched, t0_sched
        while pos <= num_iterations:
            restart_iters.add(pos)
            period = int(period * t_mult)
            pos += period

        with torch.no_grad():
            ref_encs = [
                m.encoder.encode_sequence(ref_indices) for m in self.model.models
            ]

        # Per-seq activity: frozen seqs keep historical best, stop receiving grads.
        active = np.ones(B, dtype=bool)
        no_improve = np.zeros(B, dtype=np.int32)
        sigma_patience = 0
        improve_wait = 0
        last_mean_best = float(best_scores.mean())
        opt_label = (
            f"model {generation_model_index}" if single_model else "ensemble LCB")

        for it in tqdm(range(num_iterations),
                       desc=f"  Gradient Refine ({opt_label})", leave=False):
            if not active.any():
                print(f"\n    All sequences frozen at iter {it}; stopping.")
                break

            optimizer.zero_grad()
            tau = max(0.3, temperature * (1 - 0.5 * it / num_iterations))
            seq_oh = F.gumbel_softmax(batch_logits, tau=tau, hard=True, dim=-1)

            with torch.backends.cudnn.flags(enabled=False):
                if single_model:
                    opt_pairs = [(
                        self.model.models[generation_model_index],
                        ref_encs[generation_model_index],
                    )]
                else:
                    opt_pairs = list(zip(self.model.models, ref_encs))

                fitness_list = []
                emb_list = []
                for model, ref_enc in opt_pairs:
                    model.eval()
                    seq_enc = model.encoder.encode_sequence(seq_oh)
                    if use_diversity:
                        emb_list.append(seq_enc)
                    fitness_list.append(
                        self._rank_fitness(model, seq_enc, ref_enc, T))

                fitness_stack = torch.stack(fitness_list)  # [K_opt, B]
                if single_model:
                    total_loss = -fitness_stack[0]
                else:
                    mu = fitness_stack.mean(dim=0)
                    if K > 1:
                        sigma = (fitness_stack.var(dim=0, correction=0) + 1e-8).sqrt()
                    else:
                        sigma = torch.zeros(B, device=self.device)
                    total_loss = -(mu - beta * sigma)

                if use_diversity:
                    # Frozen seqs still contribute as fixed repulsion anchors.
                    emb = torch.stack(emb_list).mean(dim=0)  # [B, H]
                    emb_n = F.normalize(emb, dim=1)
                    cos = emb_n @ emb_n.t()
                    eye = torch.eye(B, dtype=torch.bool, device=self.device)
                    mean_sim = cos.masked_fill(eye, 0.0).sum(dim=1) / max(B - 1, 1)
                    total_loss = total_loss + alpha * mean_sim

                # Soft T-content hinge: push soft T fraction up to threshold.
                if use_t_constraint:
                    t_frac = seq_oh[:, :, t_base_idx].mean(dim=1)  # [B]
                    t_deficit = F.relu(self.skip_t_threshold - t_frac)
                    total_loss = total_loss + t_loss_weight * t_deficit

                # Only active sequences contribute to the summed loss / grads.
                active_t = torch.as_tensor(
                    active, device=self.device, dtype=total_loss.dtype)
                (total_loss * active_t).sum().backward()

            # Belt-and-suspenders: zero grads on frozen rows.
            if batch_logits.grad is not None and not active.all():
                batch_logits.grad[~torch.as_tensor(active, device=self.device)] = 0

            grad_norm = nn.utils.clip_grad_norm_([batch_logits], 5.0)
            optimizer.step()
            scheduler.step()

            # ── discrete tracking with full ensemble ──
            with torch.no_grad():
                cur_seqs = self._decode_logits(batch_logits)
                batch_idx = torch.stack(
                    [sequence_to_indices(s) for s in cur_seqs]
                ).to(self.device)

                cur_ms = np.zeros((B, K), dtype=np.float64)
                for k, (model, ref_enc) in enumerate(
                        zip(self.model.models, ref_encs)):
                    se = model.encoder.encode_sequence(batch_idx)
                    cur_ms[:, k] = self._rank_fitness(
                        model, se, ref_enc, T).cpu().numpy()

                cur_means = cur_ms.mean(axis=1)
                for i in range(B):
                    if not active[i]:
                        continue
                    select = (
                        cur_ms[i, generation_model_index]
                        if single_model else cur_means[i])
                    cur_t_ok = self._meets_t_threshold(cur_seqs[i])
                    best_t_ok = self._meets_t_threshold(best_seqs[i])
                    # Prioritize sequences that meet the T threshold.
                    if use_t_constraint and cur_t_ok and not best_t_ok:
                        accept = True
                    elif use_t_constraint and not cur_t_ok and best_t_ok:
                        accept = False
                    else:
                        accept = select > best_select[i] + improve_eps

                    if accept:
                        best_select[i] = select
                        best_scores[i] = float(cur_means[i])
                        best_seqs[i] = cur_seqs[i]
                        best_model_scores[i] = cur_ms[i]
                        no_improve[i] = 0
                    else:
                        no_improve[i] += 1
                        if use_freeze and no_improve[i] >= freeze_patience:
                            active[i] = False
                            # Snap logits back to historical best discrete seq.
                            best_idx = sequence_to_indices(best_seqs[i]).to(self.device)
                            best_oh = F.one_hot(
                                best_idx, num_classes=VOCAB_SIZE).float()
                            batch_logits.data[i] = best_oh * init_scale

                mean_sigma = float(best_model_scores.std(axis=1).mean())
                self.score_history.append({
                    'step': it + 1,
                    'best': float(best_scores.max()),
                    'mean': float(best_scores.mean()),
                    'scores': best_scores.tolist(),
                    'sigma': best_model_scores.std(axis=1).tolist(),
                    'n_active': int(active.sum()),
                    **self._compute_diversity(best_seqs),
                })

            if not active.any():
                print(f"\n    All sequences frozen at iter {it + 1}; stopping.")
                break

            # ── population fitness plateau early-stop ──
            cur_mean_best = float(best_scores.mean())
            if use_improve_stop:
                if cur_mean_best > last_mean_best + improve_eps:
                    improve_wait = 0
                    last_mean_best = cur_mean_best
                else:
                    improve_wait += 1
                    if improve_wait >= improve_patience:
                        print(f"\n    Early stop at iter {it + 1}: "
                              f"mean best fitness plateaued for "
                              f"{improve_patience} iters "
                              f"(mean={cur_mean_best:.4f})")
                        break

            if (it + 1) in restart_iters and (it + 1) < num_iterations:
                for state in optimizer.state.values():
                    if 'exp_avg' in state:
                        state['exp_avg'].zero_()
                        state['exp_avg_sq'].zero_()
                print(f"    Restart at iter {it + 1}: "
                      f"best={best_scores.max():.4f}, σ={mean_sigma:.4f}, "
                      f"active={int(active.sum())}/{B}")

            if (it + 1) % 5 == 0 or it == 0 or it == num_iterations - 1:
                lr_now = scheduler.get_last_lr()[0]
                mean_t = float(np.mean(
                    [self._t_fraction(s) for s in best_seqs]))
                print(f"    Iter {it + 1:3d}: best={best_scores.max():.4f}"
                      f", mean={best_scores.mean():.4f}"
                      f", σ={mean_sigma:.4f}"
                      f", T={mean_t:.3f}"
                      f", active={int(active.sum())}/{B}"
                      f", loss={(total_loss * active_t).sum().item():.4f}"
                      f", grad={float(grad_norm):.4f}, lr={lr_now:.4f}")

        final_sigma = float(best_model_scores.std(axis=1).mean())
        print(f"  Final scores: best={best_scores.max():.4f}, "
              f"mean={best_scores.mean():.4f}, σ={final_sigma:.4f}, "
              f"frozen={int((~active).sum())}/{B}")
        print(f"  Gradient time: {time() - t0:.1f}s")

        return [(best_seqs[i], float(best_scores[i])) for i in range(B)]

    # ======================= Generate =========================
    def generate(
        self,
        init_type: str = 'topk',
        num_sequences: int = 1000,
        num_iterations: int = 100,
        learning_rate: float = 0.5,
        temperature: float = 1.0,
        alpha: float = 0.0,
        beta: float = 1.0,
        improve_patience: int = 20,
        freeze_patience: int = 15,
        generation_model_index: int = None,
    ):
        """Run the gradient generation pipeline: init → gradient refine."""
        self.model.eval()
        seqs = self.initialize_sequences(init_type, num_sequences)

        opt_seqs = self.gradient_refine(
            seqs=seqs,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            temperature=temperature,
            alpha=alpha,
            beta=beta,
            improve_patience=improve_patience,
            freeze_patience=freeze_patience,
            generation_model_index=generation_model_index,
        )
        opt_seqs.sort(key=lambda x: x[1], reverse=True)

        n_before = len(opt_seqs)
        filtered = self.filter_candidates(opt_seqs)
        if n_before > 0:
            print(f"  Filtered by T≥{self.skip_t_threshold}: "
                  f"{len(filtered)}/{n_before} kept")
        if filtered:
            opt_seqs = filtered
        else:
            print("  Warning: no sequences passed filters; "
                  "saving unfiltered results")

        # Re-score with the same eval-mode API used by scripts/score.py so
        # saved CSV scores match post-hoc scoring.
        ref_indices = self.ref_pool_indices[:self.top_k]
        rescored = []
        for seq, _ in opt_seqs:
            result = self.model.score_sequence(seq, ref_indices)
            rescored.append((seq, float(result['score'])))
        rescored.sort(key=lambda x: x[1], reverse=True)
        opt_seqs = rescored
        print(
            f"  Rescored {len(opt_seqs)} sequences in eval mode "
            f"against top-{self.top_k} refs "
            f"(best={opt_seqs[0][1]:.4f}, mean={np.mean([s for _, s in opt_seqs]):.4f})"
        )

        Path(self.generate_dir).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [{'sequence': seq, 'score': score} for seq, score in opt_seqs])
        if len(df) > 0:
            t_frac = df['sequence'].map(self._t_fraction)
            print(f"  Final T fraction: mean={t_frac.mean():.4f}, "
                  f"min={t_frac.min():.4f}, "
                  f"≥threshold={(t_frac >= self.skip_t_threshold).mean():.1%}")
        if generation_model_index is not None:
            file_name = f'{init_type}_a{alpha}_b{beta}_m{generation_model_index}'
        else:
            file_name = f'{init_type}_a{alpha}_b{beta}'
        df.to_csv(Path(self.generate_dir) / f'{file_name}_seqs.csv', index=False)

        hist_json = Path(self.generate_dir) / f'{file_name}_score_history.json'
        with open(hist_json, 'w') as f:
            json.dump(self.score_history, f)

        plot_score_history(
            data_path=str(hist_json),
            save_path=str(Path(self.generate_dir) / f'{file_name}_score_history.png'),
        )
        return df
