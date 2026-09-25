---
name: cnn-transformer-pipeline
description: Train, tune, evaluate, and interpret CNN and CNN+RoPE-Transformer models on regulatory DNA sequences using the cnn-transformer-pipeline repository (zihengc/cnn-transformer-pipeline, a fork of the Pfenning Lab cnn_pipeline). Use this skill whenever the work involves that repository or its files (train.py, dataset.py, models.py, scripts/explain.py, config-base-cnn_*.yaml, sweep-config.yaml), and more generally whenever someone wants to train a sequence model on ATAC-seq / ChIP-seq peaks, BED / narrowPeak / FASTA genomic intervals, predict enhancer or chromatin-accessibility signal from DNA, run a W&B hyperparameter sweep for such a model, score new sequences with a trained .h5 model, or run DeepSHAP attribution and TF-MoDISco-lite motif discovery on a genomic CNN — even if they never mention this pipeline by name.
---

# CNN-Transformer pipeline for regulatory DNA

This skill drives the `cnn-transformer-pipeline` repository: a TensorFlow 2.7 / Keras
pipeline that trains 1D CNNs — optionally followed by a RoPE Transformer encoder — on
genomic intervals, and then evaluates and interprets them.

Everything the pipeline does is driven by a single YAML config in a `desc`/`value`
format. Most of the work in any task is getting that config right; the commands
themselves are short. Config mistakes surface as opaque assertion errors or silent
shape mismatches deep inside TensorFlow, so validate the config before launching
anything long-running.

## Before anything else: locate the repository

The pipeline code is not bundled in this skill — it lives in its own repository so that
there is exactly one copy to maintain. Find or create it:

```bash
git clone https://github.com/zihengc/cnn-transformer-pipeline.git
cd cnn-transformer-pipeline
```

Run every command from the repository root. The demo config uses relative
`demo_data/...` paths, and `train.py` imports sibling modules (`dataset`, `models`,
`utils`) directly, so a different working directory breaks both.

Two things the repository cannot supply, which are worth confirming with the user early
rather than after a failed run:

- **An hg38 reference FASTA.** Every BED/narrowPeak input needs one, and it is too large
  to ship. Ask for its absolute path. FASTA inputs need no genome.
- **A W&B account.** `train.py` always calls `wandb.init`. `wandb login` once, or pass
  `-wandb-mode offline` (or `disabled`) when the user does not want tracking or the
  machine has no network.

## Environment

```bash
bash setup_environment.sh              # creates conda env cnn-transformer-shap-modisco
conda activate cnn-transformer-shap-modisco
```

The environment is solved for **linux-64 only** and pins TensorFlow 2.7 / Python 3.9.
On macOS or ARM it will not solve — say so plainly instead of trying to patch the YAML
into working, because the SHAP/TF-MoDISco stack is tightly version-coupled. Setup takes
10–30 minutes. GPU users must already have a driver/CUDA/cuDNN stack compatible with
TF 2.7; the environment deliberately does not install one.

## Pick the workflow

| The user wants to… | Do this |
| --- | --- |
| Check the pipeline works at all | Run the HEK293 demo (below) |
| Train on their own data | Build a config from a template → `python train.py -config <cfg>` |
| Search hyperparameters | `wandb sweep` / `scripts/sweep.sbatch` |
| Get metrics for a trained model | `scripts/validate.py` |
| Score new sequences | `scripts/get_activations.py` |
| Find motifs / explain predictions | `scripts/explain.py` or `scripts/explain_mpra.py` |

Read `references/configuration.md` before writing or editing any config — it documents
every field group, the zero-based BED column convention, and the regression vs.
classification switches. Read `references/workflows.md` for the exact command lines,
flags, and expected outputs of each workflow above, including Slurm submission. Read
`references/interpretation.md` before running DeepSHAP or TF-MoDISco-lite.
`references/troubleshooting.md` maps the pipeline's common failure messages to causes —
consult it when a run fails rather than guessing.

## The demo, as a smoke test

```bash
sed -i 's|/absolute/path/to/hg38.fa|/real/path/hg38.fa|g' example_configs/config-base-cnn_demo_template.yaml
python train.py -config example_configs/config-base-cnn_demo_template.yaml
```

The demo trains a CNN-only regression model on the included HEK293 ATAC-seq intervals
(column 6 is the continuous target) and reports held-out test metrics through the
additional-validation callback. Default is 25 epochs. For a functional check that
finishes quickly, set `num_epochs.value` to 1 first and say that is what you did —
a 1-epoch model's metrics are not meaningful.

Checkpoints `model-best.h5` and `model-last.h5` are written to the **W&B run
directory**, not the repository root. W&B prints that directory at init; capture it,
because every downstream script needs the absolute model path.

## Always validate the config first

```bash
python scripts/check_config.py <config.yaml>
```

This bundled script is dependency-light (PyYAML only, no TensorFlow), so it runs before
the environment is even active. It catches the mistakes that actually happen here:
leftover `/absolute/path/to/...` placeholders, missing data files, `*_data_paths` and
`*_targets` lists of different lengths, too few per-layer values for `num_conv_layers`,
regression configs left on a classification loss, and transformer layers requested with
`use_transformer: false`. A few seconds here saves a job that would otherwise die
minutes in, or worse, on a cluster queue hours later.

## Working habits that matter here

**Copy templates, never edit them in place.** `example_configs/*_template.yaml` are
tracked files that other runs and the sweep config point at. Write new configs to a
separate directory (e.g. `configs/`) so the user's experiments stay reproducible and
`git status` stays readable.

**Keep path and target lists aligned.** `train_data_paths[i]` pairs with
`train_targets[i]`; `utils.validate_config` asserts equal lengths but cannot tell you
that the pairing is semantically wrong. When adding a negative-set partition, add its
target entry in the same edit.

**Record what produced a model.** For anything the user may publish, note the config
path, W&B run ID, hg38 FASTA source, and GPU/CUDA versions alongside the checkpoint.
`dataset.py` seeds its sampling at 0, so the config plus the FASTA is most of
reproducibility — but only if it was written down.

**Be honest about runtime.** Full training and DeepSHAP are slow on CPU and depend on
`shap_num_bg`/`shap_num_fg` and model size. When the user wants a quick check, shrink
those knobs explicitly and tell them the result is a functional test, not a measurement.
