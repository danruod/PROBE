"""
Unified trainers for pairwise / pointwise cross-validation.

TrainEngine – single-model training / eval / pairwise-prob inference.
CVTrainer   – neural models (DL sequence or LLM embedding) cross-validation.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataloader import make_pairwise_loader, make_pointwise_loader
from src.models.base import load_embeddings
from src.models.models import create_model
from src.utils.utils import resolve_checkpoint_path
from src.utils.eval import compute_cls_metrics

# ============================= TrainEngine ==================================

class TrainEngine:
    """Single-model training engine for pairwise & pointwise modes.

    Pairwise uses MarginRankingLoss (rank). Pointwise uses MSELoss.
    """

    def __init__(self, model, mode='pairwise', input_type='sequence', emb_dict=None, 
                 learning_rate=1e-4, weight_decay=1e-5,
                 max_epochs=100, patience=10, warmup_epochs=5,
                 scheduler_type='cosine', grad_clip=1.0,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.model = model.to(device)
        self.mode = mode
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.input_type = input_type
        self.emb_dict = emb_dict

        if mode == 'pairwise':
            self.criterion = nn.MarginRankingLoss(margin=0.5)
        else:
            self.criterion = nn.MSELoss()

        trainable = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {n_train:,} / {n_total:,} ({n_train/n_total*100:.1f}%)")
    
        self.optimizer = torch.optim.AdamW(
            trainable, lr=learning_rate, weight_decay=weight_decay)
        self.grad_clip = grad_clip

        if scheduler_type == 'cosine' and warmup_epochs > 0:
            warmup = LinearLR(self.optimizer, start_factor=0.01,
                              end_factor=1.0, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(
                self.optimizer, T_max=max(1, max_epochs - warmup_epochs),
                eta_min=learning_rate * 0.01)
            self.scheduler = SequentialLR(
                self.optimizer, schedulers=[warmup, cosine],
                milestones=[warmup_epochs])
            self._epoch_sched = True
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                patience=max(3, patience // 2),
                factor=0.5, min_lr=learning_rate * 0.01)
            self._epoch_sched = False

    # ── checkpoint ─────────────────────────────────────────────────────────

    def save_checkpoint(self, path):
        """Save model state dict to *path*."""
        if path is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(path))
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load model state dict from *path*. Returns True on success."""
        path = Path(path)
        if not path.exists():
            print(f"  Warning: checkpoint not found: {path}")
            return False
        state = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        print(f"  Checkpoint loaded: {path}")
        return True


    # ── batch / loss ───────────────────────────────────────────────────────

    def _process_batch(self, batch, return_prob=False):
        batch = [t.to(self.device) for t in batch]
        if len(batch) == 3:
            return self.model(batch[0], batch[1], return_prob=return_prob), batch[2]
        elif len(batch) == 2:
            return self.model(batch[0]), batch[1]
        else:
            raise ValueError(f"Invalid batch length: {len(batch)}")


    def _compute_loss(self, outputs, labels):
        if self.mode == 'pairwise':
            targets = labels.float() * 2 - 1
            return self.criterion(outputs[:, 0], outputs[:, 1], targets)
        else:
            return self.criterion(outputs, labels.float())

    # ── train ──────────────────────────────────────────────────────────────

    def train(self, train_loader, val_loader, checkpoint_path=None):
        """Train with early stopping. Best model is loaded in-place.

        If *checkpoint_path* is provided the best model state is also
        persisted to disk so it can be resumed later.
        """
        maximize = (self.mode == 'pairwise')
        best_val = 0.0 if maximize else float('inf')
        best_state = None
        patience_ctr = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            loss_sum = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                outputs, labels = self._process_batch(batch)
                loss = self._compute_loss(outputs, labels)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                loss_sum += loss.item()
            train_loss = loss_sum / len(train_loader)

            train_metrics = self.evaluate(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            val_track = val_metrics.get('auc', 0.0) if maximize else val_loss

            lr = self.optimizer.param_groups[0]['lr']
            if maximize:
                print(f"  Ep {epoch+1}/{self.max_epochs}  "
                      f"loss={train_loss:.4f}  train_auc={train_metrics.get('auc', 0):.4f}  "
                      f"val_loss={val_loss:.4f}  val_auc={val_metrics.get('auc', 0):.4f}  "
                      f"lr={lr:.2e}")
            else:
                print(f"  Ep {epoch+1}/{self.max_epochs}  "
                      f"loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  "
                      f"val_spearman={val_metrics.get('spearman', 0):.4f}  "
                      f"lr={lr:.2e}")

            if self._epoch_sched:
                self.scheduler.step()
            else:
                self.scheduler.step(val_loss)

            improved = ((val_track > best_val + 1e-4) if maximize
                        else (val_track < best_val - 1e-4))
            if improved:
                best_val = val_track
                patience_ctr = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    metric_name = 'auc' if maximize else 'loss'
                    print(f"  Early stop at ep {epoch+1} "
                          f"(best {metric_name}={best_val:.4f})")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.save_checkpoint(checkpoint_path)
        return self.model

    # ──────────────────── Inference ─────────────────────────────────
    def predict_probs(self, test_df, batch_size=1024):
        self.model.eval()

        if self.mode == "pointwise":
            scores = []
            seqs = pd.concat([
                test_df[["s1"]].rename(columns={"s1": "seq"}),
                test_df[["s2"]].rename(columns={"s2": "seq"}),
            ]).drop_duplicates(subset="seq")["seq"].tolist()
            loader = make_pointwise_loader(
                seqs, [0.0] * len(seqs), self.input_type, self.emb_dict,
                batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for batch in loader:
                    outputs, _ = self._process_batch(batch, return_prob=True)
                    scores.extend(outputs.cpu().numpy())
            score_map = dict(zip(seqs, scores))
            probs = 1.0 / (1.0 + np.exp(-(test_df["s1"].map(score_map).values - test_df["s2"].map(score_map).values)))
            return probs

        else:
            probs = []
            loader = make_pairwise_loader(test_df, self.input_type, self.emb_dict,
                                          batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for batch in loader:
                    outputs, _ = self._process_batch(batch, return_prob=True)
                    probs.extend(outputs.cpu().numpy())
            return np.asarray(probs, dtype=np.float64)

    # ── evaluate ───────────────────────────────────────────────────────────

    def evaluate(self, dataloader):
        """Evaluate model. Returns metrics dict (cls or reg based on mode)."""
        self.model.eval()
        all_preds, all_targets, all_probs = [], [], []
        loss_sum = 0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs, labels = self._process_batch(batch, return_prob=False)
                loss = self._compute_loss(outputs, labels)
                loss_sum += loss.item()

                probs = 1.0 / (1.0 + np.exp(-(outputs[:, 0].cpu().numpy() - outputs[:, 1].cpu().numpy())))
                    
                if len(batch) == 3:
                    all_probs.extend(probs)
                    all_preds.extend((probs > 0.5))
                else:
                    all_preds.extend(outputs.cpu().numpy().ravel())
                all_targets.extend(labels.cpu().numpy().ravel())

        if len(batch) == 3:
            metrics = compute_cls_metrics(all_targets, all_preds, all_probs)
        
        metrics['loss'] = loss_sum / len(dataloader)

        return metrics

# ============================= CVTrainer ==============================

class CVTrainer:
    """CV trainer for neural models """
    
    def __init__(self, data_name, mode, model_names, folds_pairs, folds_seqs, 
                 output_dir, local_pairing, random_state, args):
        self.data_name = data_name
        self.mode = mode
        self.model_names = model_names
        self.folds_pairs = folds_pairs
        self.folds_seqs = folds_seqs

        self.output_dir = Path(output_dir)
        self.local_pairing = local_pairing
        self.random_state = random_state
        self.args = args
        
        self.result_file = self.output_dir / f'results.csv'
        self.summary_file = self.output_dir / f'summary.csv'

    # ── setup input type and embedding ──────────────────────────────────────

    def _device(self):
        gpu = getattr(self.args, 'gpu_id', None)
        if gpu is not None and torch.cuda.is_available():
            return f'cuda:{gpu}'
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _setup_input_type_and_embedding(self, model_name):
        """
        Setup input type and embedding.
        input_type:
            'sequence'  – DL encoders on raw DNA sequences (one-hot).
            'embedding' – linear probe on precomputed LLM embeddings.
        """
        if model_name in ['alphagenome', 'evo2']:
            emb_path = f'{self.args.data_output_dir}/{self.data_name}/embedding/'
            self.emb_dict, self.emb_dim = load_embeddings(emb_path, model_name)
            self.input_type = 'embedding'
        else:
            self.emb_dict = None
            self.emb_dim = self.args.vocab_size
            self.input_type = 'sequence'

    # ── result I/O ─────────────────────────────────────────────────────────

    def save_fold_result(self, fold_id, fold_results):
        rows = []
        for name, metrics in fold_results.items():
            if self.mode == 'pairwise':
                mode = 'local' if self.local_pairing else 'global'
            else:
                mode = self.mode
            row = {'fold': fold_id, 'data_name': self.data_name, 
                   'split_method': self.args.split_method,
                   'name': str(self.args.max_n_seqs),
                   'model': name, 'mode': mode, 'lr': self.args.learning_rate,
                   }
            row.update(metrics)
            rows.append(row)

        df = pd.DataFrame(rows)
        self.result_file.parent.mkdir(parents=True, exist_ok=True)
        header = not self.result_file.exists()
        df.to_csv(self.result_file, mode='a', header=header, index=False)
        print(f"Fold {fold_id} results saved to: {self.result_file}")
        return df

    def save_cv_summary(self, results):
        print(f"\n{'='*60}")
        print(f"CV Summary")
        print(f"{'='*60}")

        metric_cols = [c for c in results.columns
                       if results[c].dtype == 'float64' and c != 'lr']
        non_metric = [c for c in results.columns
                      if c not in metric_cols and c != 'fold']

        summary = results.groupby(non_metric, dropna=False).agg(
            {m: 'mean' for m in metric_cols}).reset_index()
        self.summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(self.summary_file, mode='a', index=False)
        print(f"{summary.T}")
        print(f"Summary saved to: {self.summary_file}")
        return summary

    # ── dataloaders ────────────────────────────────────────────────────────

    def _make_pairwise_loader(self, data, batch_size, shuffle=True):
        return make_pairwise_loader(
            data, self.input_type, self.emb_dict,
            batch_size=batch_size, shuffle=shuffle,
            mutation_rate=getattr(self.args, 'mutation_rate', 0.0),
            num_workers=getattr(self.args, 'num_workers', 0))

    def _make_pointwise_loader(self, seqs, scores, batch_size, shuffle=True):
        return make_pointwise_loader(
            seqs, scores, self.input_type, self.emb_dict,
            batch_size=batch_size, shuffle=shuffle,
            num_workers=getattr(self.args, 'num_workers', 0))

    # ── checkpoint path ────────────────────────────────────────────────────

    def _checkpoint_path(self, model_name, fold_id):
        """Deterministic checkpoint path for a (model, fold, lr) combo."""
        return resolve_checkpoint_path(
            data_name=self.data_name,
            split_method=self.args.split_method,
            max_n_seqs=self.args.max_n_seqs,
            model_name=model_name,
            lr=self.args.learning_rate,
            fold_id=fold_id,
            mode=self.mode,
            local_pairing=bool(self.local_pairing),
        )

    # ── single fold ────────────────────────────────────────────────────────

    def train_fold(
        self, model_name, fold_id, train_ld, val_ld, test_pair_ld
    ):
        torch.manual_seed(self.random_state + fold_id)
        np.random.seed(self.random_state + fold_id)

        device = self._device()
        model = create_model(
            model_name,
            input_type=self.input_type,
            mode=self.mode,
            hidden_dim=self.args.hidden_dim,
            dropout=self.args.dropout,
            emb_dim=self.emb_dim,
            vocab_size=self.args.vocab_size,
        )

        ckpt = self._checkpoint_path(model_name, fold_id)
        if self.args.resume and self.args.resume_path is not None:
            ckpt = Path(self.args.resume_path)

        if self.args.skip_training and not ckpt.exists():
            raise FileNotFoundError(
                f"Prediction-only mode requires an existing checkpoint: {ckpt}"
            )

        engine = TrainEngine(
            model=model, mode=self.mode,
            input_type=self.input_type,
            emb_dict=self.emb_dict,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            max_epochs=(self.args.max_epochs
                        if not getattr(self.args, 'debug', False) else 2),
            patience=self.args.patience,
            warmup_epochs=self.args.warmup_epochs,
            scheduler_type=self.args.scheduler_type,
            grad_clip=self.args.grad_clip,
            device=device,
        )

        if self.args.skip_training:
            engine.load_checkpoint(ckpt)
            print(f"  Skipping training (checkpoint loaded)")
        else:
            if self.args.resume and ckpt.exists():
                engine.load_checkpoint(ckpt)
                print(f"  Resumed weights from checkpoint; continuing training")
            engine.train(train_ld, val_ld, checkpoint_path=ckpt)

        test_metrics = {}

        m = engine.evaluate(test_pair_ld)
        for k, v in m.items():
            print(f'  {k}: {v:.4f}')
            test_metrics[k] = v

        return self.save_fold_result(fold_id, {model_name: test_metrics})

    # ── CV loop ────────────────────────────────────────────────────────────

    def train_cv(self):
        print(f"\n{'*'*60}")
        print(f"CVTrainer  mode={self.mode}  Models: {set(self.model_names)}")

        lr_list = (self.args.learning_rate
                   if isinstance(self.args.learning_rate, list)
                   else [self.args.learning_rate])

        all_results = pd.DataFrame()
        for i, model_name in enumerate(self.model_names):

            self._setup_input_type_and_embedding(model_name)

            print(f"\n{'='*60}")
            print(f"Training model: {i+1}/{len(self.model_names)} {model_name}")

            for fold_id in range(len(self.folds_pairs)):
                fold_data = self.folds_pairs[fold_id]
                train_data, val_data, test_data = fold_data[0], fold_data[1], fold_data[2]

                fold_mask = self.folds_seqs['fold'] == fold_id
                non_test_seqs = self.folds_seqs.loc[~fold_mask, 'seq'].tolist()
                seq_to_val = dict(zip(self.folds_seqs['seq'], self.folds_seqs['value']))

                test_pair_ld = self._make_pairwise_loader(
                    test_data, self.args.test_batch_size, shuffle=False)

                if self.mode == 'pointwise':
                    val_size = self.args.val_size

                    rng = np.random.RandomState(self.random_state + fold_id)
                    n_val = max(1, int(len(non_test_seqs) * val_size))
                    perm = rng.permutation(len(non_test_seqs))

                    val_seqs = [non_test_seqs[i] for i in perm[:n_val]]
                    train_seqs = [non_test_seqs[i] for i in perm[n_val:]]
                    tr_scores = [seq_to_val[s] for s in train_seqs]
                    va_scores = [seq_to_val[s] for s in val_seqs]

                    print(f"  Pointwise: {len(train_seqs)} train, {len(val_seqs)} val seqs")
                    train_ld = self._make_pointwise_loader(
                        train_seqs, tr_scores, self.args.train_batch_size, shuffle=True)
                    val_ld = self._make_pointwise_loader(
                        val_seqs, va_scores, self.args.val_batch_size, shuffle=False)
                else:
                    train_ld = self._make_pairwise_loader(
                        train_data, self.args.train_batch_size, shuffle=True)
                    val_ld = self._make_pairwise_loader(
                        val_data, self.args.val_batch_size, shuffle=False)

                for lr in lr_list:
                    self.args.learning_rate = float(lr)
                    print(f"\n{'*'*30}")
                    print(f" Fold {fold_id} lr={lr}")
                    print(f"{'*'*30}")
                    df = self.train_fold(
                        model_name, fold_id, train_ld, val_ld, test_pair_ld
                    )
                    all_results = pd.concat([all_results, df],
                                            ignore_index=True)

        self.save_cv_summary(all_results)

