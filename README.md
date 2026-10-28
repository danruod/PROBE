# PROBE

PROBE learns a pairwise ranking model over DNA aptamer sequences and optimizes new candidates with gradient-based refinement against an ensemble of cross-validation folds.


## Google Colab

<a href="https://colab.research.google.com/drive/1qNCL2jkttn2GBzFilPktAfDupKTHW5Tz?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> <br />

We prepared a convenient Google Colab notebook to test PROBE's code functionalities (data → train → generate → score). The Colab configuration is intentionally small and is only meant as a quick smoke test. For real runs, choose parameters according to your dataset scale and observed model performance (e.g. number of folds, sequence budget, training epochs, and generation iterations), and prefer a local or cluster installation with adequate GPU resources.


## Features

- **Data preprocessing** – k-mer (or random) CV splits and local/global ranking pairs
- **Training** – pairwise models (`cnn`, `lstm`, `transformer`, hybrids)
- **Generation** – `freq_prior` / `topk` / `random` init, Gumbel-Softmax refine with diversity (`alpha`) and uncertainty (`beta`) penalties
- **Scoring** – eval-mode ensemble fitness against a top-k enriched reference pool (same metric as generation)

## Repository layout

```text
PROBE/
├── config/                   # default YAML (data / train / generate)
├── scripts/
│   ├── data.py               # build CV pair datasets
│   ├── train.py              # cross-validation training
│   ├── generate.py           # sequence generation
│   └── score.py              # score sequences with a trained ensemble
├── src/
│   ├── data/                 # preprocessing + dataloaders
│   ├── models/               # encoders, EnsembleModel, trainers
│   ├── generation/           # PROBEGenerator
│   └── utils/                # config helpers, eval, plotting
├── notebooks/
│   └── demo.csv              # small demo table (seq, value)
├── environment.yml           # conda env (includes CUDA torch)
├── requirements.txt          # pip deps (install torch separately)
└── run.sh                    # optional one-liner helper
```

Scripts create local runtime folders (`dataset/`, `checkpoints/`, `experiments/`, `logs/`). 

Optional per-dataset overrides: `config/<data_name>/{data,train,generate}_config.yaml`.

## Input data format

Pass any CSV via `--data_path`. Required columns:

| column  | description |
|---------|-------------|
| `seq`   | DNA sequence (`A/T/G/C`; `N` treated as unknown/pad) |
| `value` | continuous enrichment / affinity score |

Example:

```csv
seq,value
GGGGAGAAGGACGACCCCGAGCGGGAAGAGTGCACTGGAA,1682
GGACCCGCAAGGGAGGGAGAGGGGGAACAAGGGAAGTGTG,1525
```

A small demo file is included at [`notebooks/demo.csv`](notebooks/demo.csv).

If `--data_name` is omitted, it defaults to the CSV stem
(e.g. `notebooks/demo.csv` → `demo`). Outputs and checkpoints are keyed by `data_name`.

## Installation

```bash
# create env
conda create -n probe python=3.11 pip -y
conda activate probe

# install CUDA PyTorch (match your CUDA; example: cu124)
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# install remaining deps
pip install -r requirements.txt
```

Or:

```bash
conda env create -f environment.yml
conda activate probe
```

## Quick start

Run all commands from the repository root.

### 1. Prepare data

```bash
python scripts/data.py --data_path notebooks/demo.csv
# equivalent:
# python scripts/data.py --data_path notebooks/demo.csv --data_name demo
```

Writes fold assignments and train/val/test pair CSVs under
`dataset/<data_name>/<split_method>/`.

Configs load from `config/<data_name>/data_config.yaml` when present, else
`config/data_config.yaml`.

### 2. Train

Edit `config/train_config.yaml` (or `config/<data_name>/train_config.yaml`):

- `training.skip_training: false` to train (not only evaluate)
- `model.models`, `learning_rate`, `max_epochs`, etc.

```bash
python scripts/train.py --data_path notebooks/demo.csv --models cnn
```

Checkpoints are saved under:

```text
checkpoints/<data_name>/<split_method>/<max_n_seqs>/
```

### 3. Generate

Initialization defaults to `freq_prior` (legacy alias: `ref_content`).
If `ckpt_dir` is omitted, it is inferred as
`checkpoints/<data_name>/<split>/<max_n_seqs>/`.

Fitness uses the top-`top_k` enriched references (default `20`). Saved CSV
scores are written in **eval mode** so they match `scripts/score.py`.

```bash
python scripts/generate.py --data_path notebooks/demo.csv --model_name cnn
```

Outputs go to `experiments/generation/<data_name>_<models>/<init>_<n>/`
(`*_seqs.csv`, score history JSON/PNG).

### 4. Score (optional)

```bash
python scripts/score.py \
  --seq_file experiments/generation/.../freq_prior_a0.1_b0.05_seqs.csv \
  --data_path notebooks/demo.csv \
  --model_name cnn \
  --top_k 20
```

The sequence CSV should contain a `seq` / `sequence` / `gen_seq` column.
An optional `ref_seq` column scores each row against its own reference;
otherwise a top-k enriched pool is used (same pool logic as generation).


<!-- ## Citation

If you use PROBE in your research, please cite this repository (citation entry TBD). -->

