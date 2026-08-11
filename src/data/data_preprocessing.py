#!/usr/bin/env python3
"""
Data preprocessing for ranking-pair CV folds.

Pipeline:
1. Load sequence/value table
2. Build group K-fold splits (random or k-mer family)
3. Build train / val / test pair CSVs per fold
"""

import itertools
import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, pdist

warnings.filterwarnings('ignore')


def _uniform_sequence_length(seqs):
    """Return shared length if all sequences have the same length, else None."""
    if not seqs:
        return 0
    lo = hi = len(seqs[0])
    for s in seqs[1:]:
        n = len(s)
        if n < lo:
            lo = n
        if n > hi:
            hi = n
    return lo if lo == hi else None


def _fill_char_matrix_rows(sequences, width):
    """Pad/truncate to `width` with ord('N'); int8 matrix for Hamming distance."""
    n = len(sequences)
    m = np.full((n, width), ord('N'), dtype=np.int8)
    for i, seq in enumerate(sequences):
        raw = str(seq).encode('latin-1', errors='ignore')
        L = min(len(raw), width)
        if L > 0:
            m[i, :L] = np.frombuffer(raw[:L], dtype=np.uint8).astype(np.int8)
    return m


def pairwise_edit_distance_matrix_cross(seq_a, seq_b):
    """
    Rectangular (n_a x n_b) distances between two sequence lists.
    Uses Hamming when both sides are equal-length; otherwise Levenshtein.
    """
    a = [str(s) for s in seq_a]
    b = [str(s) for s in seq_b]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.zeros((na, nb), dtype=np.float64)

    ua, ub = _uniform_sequence_length(a), _uniform_sequence_length(b)
    if ua is not None and ub is not None and ua == ub and ua > 0:
        return cdist(_fill_char_matrix_rows(a, ua),
                     _fill_char_matrix_rows(b, ub),
                     metric='hamming') * ua

    import rapidfuzz
    print(f"  [distance] variable-length batch, rapidfuzz {na}x{nb}")
    return rapidfuzz.process.cdist(
        a, b, scorer=rapidfuzz.distance.Levenshtein.distance, workers=-1
    ).astype(np.float64)


class DataPreprocessor:
    """Build CV splits and ranking pairs from a sequence/value table."""

    def __init__(
        self,
        data_path=None,
        data_name=None,
        raw_data_dir='./dataset',
        data_output_dir='./dataset',
        random_state=42,
        n_splits=5,
        split_method='kmer',
        max_n_seqs=10,
        vocab_size=4,
        load_raw_data=True,
        min_value=None,
        log_transform=False,
    ):
        from src.utils.utils import resolve_data_identity

        self.split_method = split_method
        self.n_splits = int(n_splits)
        self.max_n_seqs = float(max_n_seqs)
        self.random_state = random_state
        self.log_transform = log_transform
        self.vocab_size = vocab_size

        self.data_path, self.data_name = resolve_data_identity(
            data_path=data_path,
            data_name=data_name,
            data_output_dir=data_output_dir,
            raw_data_dir=raw_data_dir,
        )

        np.random.seed(self.random_state)

        self.df = None
        self.seqs_set = None
        self.fold_assignments = None

        if load_raw_data:
            self.load_raw_data(log_transform=log_transform, min_value=min_value)
            self.seqs_set = set(self.df['seq'].astype(str))

        # Pair CSVs live under {out}/{name}/{split_method}/
        self.data_output_dir = Path(data_output_dir) / self.data_name / f'{self.split_method}'

    # ------------------------------------------------------------------ load
    def load_raw_data(self, log_transform=None, min_value=None):
        """Load raw CSV (``self.data_path``) with columns seq, value."""
        log_transform = self.log_transform if log_transform is None else log_transform
        data_path = Path(self.data_path)

        if not data_path.exists():
            raise FileNotFoundError(f'Raw data not found: {data_path}')

        raw = pd.read_csv(data_path)
        cols = {c.lower(): c for c in raw.columns}
        if 'seq' in cols and 'value' in cols:
            self.df = raw[[cols['seq'], cols['value']]].copy()
        elif len(raw.columns) >= 2 and 'seq' not in cols:
            # Fallback: first two columns.
            self.df = raw.iloc[:, :2].copy()
        else:
            raise ValueError(f'Need seq/value columns, got: {list(raw.columns)}')

        # check if the df has 'seq' and 'value' columns
        if 'seq' not in self.df.columns or 'value' not in self.df.columns:
            raise ValueError(f'Need seq/value columns, got: {list(self.df.columns)}')

        self.df['seq'] = self.df['seq'].astype(str).str.strip()
        self.df['value'] = self.df['value'].astype(np.float64)
        self.df = self.df.drop_duplicates(subset=['seq'], keep='first').reset_index(drop=True)
        print(f'Loaded {len(self.df)} sequences from {data_path}')

        if min_value is not None:
            before = len(self.df)
            self.df = self.df[self.df['value'] >= float(min_value)].reset_index(drop=True)
            print(f'Filtered value < {min_value}: {before} -> {len(self.df)}')

        if log_transform:
            # Avoid log2(0).
            vals = self.df['value'].values
            if np.any(vals <= 0):
                vals = vals + 1e-8
                self.df['value'] = vals
            self.df['value'] = np.log2(self.df['value'])
            print(f'Applied log2 transform; value range '
                  f'[{self.df["value"].min():.4e}, {self.df["value"].max():.4e}]')

        seq_chars = set()
        for seq in self.df['seq']:
            for char in seq:
                seq_chars.add(char)
        # remove N
        seq_chars = seq_chars - {'N'}
        if self.vocab_size != len(seq_chars):
            print(f'Warning: vocabulary in the {self.data_name} includes {seq_chars}, but the vocab_size is set to {self.vocab_size}')

        return self.df

    def find_best_sequences(self, top_k=10, seq_length=None):
        """Return top-k sequences by value, optionally filtered by length."""
        if self.df is None:
            self.load_raw_data()
        df = self.df
        if seq_length is not None:
            df = df[df['seq'].str.len() == seq_length]
        return df.sort_values(by='value', ascending=False).head(top_k)

    # ---------------------------------------------------------------- splits
    def split_by_random(self, sequences):
        """Random contiguous fold assignment after shuffle."""
        print(f'\nRandom {self.n_splits}-fold split...')
        rng = np.random.RandomState(self.random_state)
        n = len(sequences)
        indices = rng.permutation(n)
        fold_assignments = np.empty(n, dtype=int)
        # np.array_split keeps fold sizes balanced when n % n_splits != 0.
        for fold_id, chunk in enumerate(np.array_split(indices, self.n_splits)):
            fold_assignments[chunk] = fold_id
        for fold_id in range(self.n_splits):
            print(f'  Fold {fold_id}: {int((fold_assignments == fold_id).sum())} sequences')
        return fold_assignments

    def split_by_kmer(self, sequences, kmer_k=4, init_threshold=0.4, min_threshold=0.005):
        """
        Group K-Fold from k-mer composition families.

        1. L2-normalized k-mer count vectors
        2. Average-linkage hierarchical clustering on cosine distance
        3. Recursively tighten the cut until every cluster is < n // n_splits
           (random shard only as a last resort)
        4. LPT bin-pack clusters into n_splits balanced folds
        """
        print(f'\nK-mer Group {self.n_splits}-Fold split...')
        n_total = len(sequences)
        if n_total == 0:
            return np.array([], dtype=int)

        max_cluster_size = max(1, n_total // self.n_splits)
        print(f'  n={n_total}, max cluster size={max_cluster_size}')

        vocab = [''.join(p) for p in itertools.product('ACGT', repeat=kmer_k)]
        vocab_index = {m: i for i, m in enumerate(vocab)}
        dim = len(vocab)

        def _kmer_vec(seq):
            counts = np.zeros(dim, dtype=np.float64)
            s = seq.upper().replace('U', 'T')
            for i in range(len(s) - kmer_k + 1):
                km = s[i:i + kmer_k]
                j = vocab_index.get(km)
                if j is not None:
                    counts[j] += 1.0
            norm = np.linalg.norm(counts)
            return counts / norm if norm > 1e-12 else counts

        X = np.stack([_kmer_vec(s) for s in sequences])
        shard_rng = np.random.RandomState(self.random_state)

        def _hcluster(indices, threshold):
            if len(indices) <= 1:
                return [list(indices)]
            condensed = pdist(X[indices], metric='cosine')
            # All-identical vectors -> condensed may contain NaN; treat as distance 0.
            condensed = np.nan_to_num(condensed, nan=0.0)
            Z = linkage(condensed, method='average')
            labels = fcluster(Z, t=threshold, criterion='distance')
            groups = defaultdict(list)
            for i, cid in enumerate(labels):
                groups[cid].append(indices[i])
            return list(groups.values())

        def _split_oversized(indices, threshold):
            sub_clusters = _hcluster(indices, threshold)
            # If clustering did not split and we can still tighten, force a cut.
            if (len(sub_clusters) == 1
                    and len(indices) > max_cluster_size
                    and threshold * 0.5 >= min_threshold):
                return _split_oversized(indices, threshold * 0.5)

            result = []
            for sc in sub_clusters:
                if len(sc) <= max_cluster_size:
                    result.append(sc)
                    continue
                tighter = threshold * 0.5
                if tighter >= min_threshold:
                    result.extend(_split_oversized(sc, tighter))
                    continue
                # Last resort: random shard (breaks family purity; logged above).
                arr = list(sc)
                shard_rng.shuffle(arr)
                for start in range(0, len(arr), max_cluster_size):
                    result.append(arr[start:start + max_cluster_size])
            return result

        clusters = _split_oversized(list(range(n_total)), init_threshold)
        sizes = sorted((len(c) for c in clusters), reverse=True)
        print(f'  clusters={len(clusters)}, top sizes={sizes[:10]}, '
              f'max={sizes[0]} (cap {max_cluster_size})')

        clusters.sort(key=len, reverse=True)
        bins = [[] for _ in range(self.n_splits)]
        bin_sizes = np.zeros(self.n_splits, dtype=int)
        for cluster in clusters:
            b = int(np.argmin(bin_sizes))
            bins[b].extend(cluster)
            bin_sizes[b] += len(cluster)

        fold_assignments = np.empty(n_total, dtype=int)
        for fold_id, group_indices in enumerate(bins):
            fold_assignments[group_indices] = fold_id

        assert int(bin_sizes.sum()) == n_total
        for fold_id in range(self.n_splits):
            print(f'  Fold {fold_id}: {int(bin_sizes[fold_id])} sequences')
        imbalance = (bin_sizes.max() - bin_sizes.min()) / max(n_total, 1) * 100
        print(f'  max-min imbalance: {imbalance:.1f}%')
        return fold_assignments

    def _load_fold_assignments(self):
        """Load or create `{n_splits}fold_splits.csv` under the method directory."""
        splits_path = self.data_output_dir / f'{self.n_splits}fold_splits.csv'
        if splits_path.exists():
            print(f'Loading existing splits: {splits_path}')
            self.fold_assignments = pd.read_csv(splits_path)
            return

        if self.seqs_set is None:
            if self.df is None:
                self.load_raw_data()
            self.seqs_set = set(self.df['seq'].astype(str))

        sequences = sorted(self.seqs_set)
        if self.split_method == 'random':
            folds = self.split_by_random(sequences)
        elif self.split_method == 'kmer':
            folds = self.split_by_kmer(sequences)
        else:
            raise ValueError(f'Invalid split_method: {self.split_method}')

        self.fold_assignments = pd.DataFrame({'seq': sequences, 'fold': folds})
        self.data_output_dir.mkdir(parents=True, exist_ok=True)
        self.fold_assignments.to_csv(splits_path, index=False)
        print(f'Saved splits: {splits_path}')

    # ----------------------------------------------------------------- pairs
    def create_pairs(self, val_size=0.1):
        """Build train / val / test pair CSVs for every fold."""
        data_path = Path(self.data_output_dir) / 'fold0_test_pairs.csv'
        if data_path.exists():
            print(f'Pairs already exist: {data_path}')
            return

        if self.df is None:
            self.load_raw_data()
        seq_to_values = dict(zip(self.df['seq'].astype(str), self.df['value']))
        self._load_fold_assignments()

        print(f'\n=== Creating pairs ({self.split_method}, {self.n_splits} folds) ===')
        for fold_id in range(self.n_splits):
            print(f'\nFold {fold_id}...')
            fa = self.fold_assignments
            test_seqs = fa.loc[fa['fold'] == fold_id, 'seq'].astype(str).tolist()
            non_test = fa.loc[fa['fold'] != fold_id, 'seq'].astype(str).tolist()

            rng = np.random.RandomState(self.random_state + fold_id)
            n_val = max(1, int(len(non_test) * val_size)) if non_test else 0
            perm = rng.permutation(len(non_test))
            val_seqs = [non_test[i] for i in perm[:n_val]]
            train_seqs = [non_test[i] for i in perm[n_val:]]

            print(f'  train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}')

            train_vals = [seq_to_values[s] for s in train_seqs]
            val_vals = [seq_to_values[s] for s in val_seqs]
            test_vals = [seq_to_values[s] for s in test_seqs]

            self.create_train_pairs(fold_id, train_seqs, train_vals, val_seqs, val_vals)
            self.create_test_pairs(fold_id, test_seqs, test_vals, train_seqs, train_vals)

    def create_train_pairs(self, fold_id, train_seqs, train_vals, val_seqs, val_vals):
        """Write fold train_local / train_global / val pair CSVs."""
        empty = pd.DataFrame(columns=['s1', 's2', 'value1', 'value2', 'label', 'split'])

        if len(train_seqs) > 1:
            print(f'### train pairs (neighbors={int(self.max_n_seqs)})')
            dist = pairwise_edit_distance_matrix_cross(train_seqs, train_seqs)
            local_pairs, global_pairs = self._create_cross_pairs(
                train_seqs, train_vals, train_seqs, train_vals,
                dist_matrix=dist,
                n_neighbors=int(self.max_n_seqs),
                return_random=True,
            )
            train_local_df = (
                pd.DataFrame(local_pairs).drop_duplicates() if local_pairs else empty.copy()
            )
            train_global_df = (
                pd.DataFrame(global_pairs).drop_duplicates() if global_pairs else empty.copy()
            )
            if len(train_local_df):
                train_local_df['split'] = 'train_local'
            if len(train_global_df):
                train_global_df['split'] = 'train'

            print(f'*** train_local={len(train_local_df)}, train_global={len(train_global_df)}')
            train_local_df = self.create_balanced_dataset(train_local_df)
            train_global_df = self.create_balanced_dataset(train_global_df)

            os.makedirs(f'{self.data_output_dir}/{self.max_n_seqs}', exist_ok=True)
            train_local_df.to_csv(
                f'{self.data_output_dir}/{self.max_n_seqs}/fold{fold_id}_train_local_pairs.csv', index=False)
            train_global_df.to_csv(
                f'{self.data_output_dir}/{self.max_n_seqs}/fold{fold_id}_train_global_pairs.csv', index=False)

        # Val = fixed small cross sample against train (stable eval size).
        val_n_neighbors = 10
        if val_seqs:
            print(f'### val pairs (neighbors={val_n_neighbors})')
            val_pairs, _ = self._create_cross_pairs(
                val_seqs, val_vals, train_seqs, train_vals,
                n_neighbors=val_n_neighbors,
                return_random=False,
            )
            val_df = pd.DataFrame(val_pairs).drop_duplicates() if val_pairs else empty.copy()
            if len(val_df):
                val_df['split'] = 'val'
            print(f'*** val={len(val_df)}')
            val_df = self.create_balanced_dataset(val_df)
            val_df.to_csv(
                f'{self.data_output_dir}/{self.max_n_seqs}/fold{fold_id}_val_pairs.csv', index=False)

    def create_test_pairs(self, fold_id, test_seqs, test_vals, train_seqs, train_vals,
                          n_neighbors=50):
        """Write fold test (test×train) and test_inner (test×test) pair CSVs."""
        empty = pd.DataFrame(columns=['s1', 's2', 'value1', 'value2', 'label', 'split'])
        test_df = empty.copy()

        print('### test pairs (cross, inner)')
        if test_seqs and train_seqs:
            cross_pairs, _ = self._create_cross_pairs(
                test_seqs, test_vals, train_seqs, train_vals,
                n_neighbors=n_neighbors,
                return_random=False,
            )
            if cross_pairs:
                test_df = pd.DataFrame(cross_pairs).drop_duplicates()
                test_df['split'] = 'test'

        print(f'*** test={len(test_df)}')
        test_df = self.create_balanced_dataset(test_df)
        test_df.to_csv(f'{self.data_output_dir}/fold{fold_id}_test_pairs.csv', index=False)

    def _pair_record(self, s_hi, s_lo, v_hi, v_lo):
        return {
            's1': s_hi, 's2': s_lo,
            'value1': v_hi, 'value2': v_lo,
            'label': 1, 
        }

    def _create_cross_pairs(
        self,
        left_seqs,
        left_values,
        right_seqs,
        right_values,
        n_neighbors,
        dist_matrix=None,
        return_random=False,
    ):
        """
        For each left sequence, take up to n_neighbors right neighbors by distance.
        Pairs are oriented so higher value is s1 (label=1). Equal values are skipped.
        """
        n_left, n_right = len(left_seqs), len(right_seqs)
        if n_left == 0 or n_right == 0:
            return [], []

        print(f'   cross pairs: neighbors={n_neighbors}, return_random={return_random}')

        left_values = np.asarray(left_values, dtype=np.float64)
        right_values = np.asarray(right_values, dtype=np.float64)
        if dist_matrix is None:
            print(f'   distance matrix {n_left}x{n_right}')
            dist_matrix = pairwise_edit_distance_matrix_cross(left_seqs, right_seqs)

        actual_n = min(int(n_neighbors), n_right)
        local_pairs, random_pairs = [], []
        seen_local, seen_random = set(), set()

        def _try_add(store, seen, i, j):
            if left_seqs[i] == right_seqs[j]:
                return False
            v_l, v_r = left_values[i], right_values[j]
            if v_l == v_r:
                return False
            if v_l > v_r:
                key = (left_seqs[i], right_seqs[j])
                if key in seen:
                    return False
                seen.add(key)
                store.append(self._pair_record(left_seqs[i], right_seqs[j], v_l, v_r))
            else:
                key = (right_seqs[j], left_seqs[i])
                if key in seen:
                    return False
                seen.add(key)
                store.append(self._pair_record(right_seqs[j], left_seqs[i], v_r, v_l))
            return True

        rng = np.random.RandomState(self.random_state)
        for i in range(n_left):
            order = np.argsort(dist_matrix[i], kind='mergesort')
            num = 0
            for j in order:
                if _try_add(local_pairs, seen_local, i, j):
                    num += 1
                    if num >= actual_n:
                        break
            if num < actual_n:
                print(f'Warning: only {num}/{actual_n} local pairs for {left_seqs[i]}')

            if not return_random:
                continue

            # Global / random pairs: shuffle right indices (not distance-ordered).
            num = 0
            for j in rng.permutation(n_right):
                if _try_add(random_pairs, seen_random, i, j):
                    num += 1
                    if num >= actual_n:
                        break
            if num == 0:
                print(f'Warning: no random pairs for {left_seqs[i]}')

        return local_pairs, random_pairs

    def create_balanced_dataset(self, data):
        """Flip half of the pairs (label 0) so labels are ~balanced, then shuffle."""
        if data is None or len(data) == 0:
            return pd.DataFrame(columns=['s1', 's2', 'value1', 'value2', 'label', 'split'])

        half1 = data.sample(frac=0.5, random_state=self.random_state)
        half2 = data.drop(half1.index).copy()
        half2[['s1', 's2', 'value1', 'value2']] = half2[['s2', 's1', 'value2', 'value1']].values
        half2['label'] = 0
        balanced = pd.concat([half1, half2], ignore_index=True).drop_duplicates()
        return balanced.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

    def load_all_folds_data(self, local_pairing=True):
        """Load all folds: {fold_id: (train, val, test)}."""
        all_folds = {}
        for fold_id in range(self.n_splits):
            try:
                if local_pairing:
                    train_file = f'{self.data_output_dir}/{self.max_n_seqs}/fold{fold_id}_train_local_pairs.csv'
                else:
                    train_file = f'{self.data_output_dir}/{self.max_n_seqs}/fold{fold_id}_train_global_pairs.csv'

                val_file = f'{self.data_output_dir}/{self.max_n_seqs}/fold{fold_id}_val_pairs.csv'
                test_file = f'{self.data_output_dir}/fold{fold_id}_test_pairs.csv'

                if not os.path.exists(train_file):
                    print(f'Train file missing: {train_file}')
                    train_data = None
                else:
                    print(f'Loading train: {train_file}')
                    train_data = pd.read_csv(train_file)

                val_data = pd.read_csv(val_file) if os.path.exists(val_file) else None

                if not os.path.exists(test_file):
                    raise FileNotFoundError(f'Test file missing: {test_file}')  
                test_data = pd.read_csv(test_file)
                print(f'Loading test: {test_file}')
                print(
                    f'train={len(train_data) if train_data is not None else 0}, '
                    f'val={len(val_data) if val_data is not None else 0}, '
                    f'test={len(test_data)}'
                )
                all_folds[fold_id] = train_data, val_data, test_data
            except FileNotFoundError as e:
                print(f'Warning: {e}')
                
        return all_folds
