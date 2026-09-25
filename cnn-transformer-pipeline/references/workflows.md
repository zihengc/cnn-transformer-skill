# Workflows

All commands run from the repository root with the conda environment active.

## Train

```bash
python train.py -config configs/my-run.yaml
# -wandb-mode offline    # or disabled, when there is no network / no tracking wanted
```

Output: per-epoch loss and metrics in the terminal and W&B; separate metrics for each
dataset under `additional_val_data_paths`; `model-best.h5` (lowest validation loss) and
`model-last.h5` in the W&B run directory, also uploaded to W&B.

Report the run directory to the user as soon as W&B prints it. Every later step needs
an absolute path to a checkpoint, and hunting for it afterwards is tedious.

### On Slurm

`scripts/train.sbatch` intentionally hard-codes no partition, account, QoS, or GPU
syntax — those differ per cluster and belong on the `sbatch` line:

```bash
sbatch --partition=<gpu-partition> --gres=gpu:1 \
  scripts/train.sbatch configs/my-run.yaml [ENV_NAME] [WANDB_MODE]
```

Defaults: 1 node, 1 task, 4 CPUs, 32 GB, 48 h — override any of them on the command
line. The script resolves a relative config against the submit directory, then the repo
root; it needs `mamba`/`conda` on PATH (load the cluster's conda module before
submitting) and uses `conda run` so no activation is needed inside the job. Set
`WANDB_DIR` to a writable filesystem before `sbatch` on clusters where `$HOME` is small
or read-only on compute nodes.

## Hyperparameter sweep

`example_configs/sweep-config.yaml` is a W&B random sweep over `lr_init`,
`lr_exp_decay_per_epoch`, and `num_conv_layers`, minimizing `val_mean_squared_error`.
Its `command:` block pins `-config example_configs/config-base-cnn_demo_template.yaml`
— **edit that path** when sweeping a different base config, otherwise the sweep silently
trains the demo. Swept parameters override the base config via `parse_known_args`, so
any config key can be swept by adding it under `parameters:`. For a classification
sweep, also change `metric.name` to a metric that run actually emits.

```bash
wandb sweep example_configs/sweep-config.yaml     # prints <entity>/<project>/<sweep-id>
wandb agent <entity>/<project>/<sweep-id>
```

On Slurm, one trial per array task, at most two GPUs at a time:

```bash
sbatch scripts/sweep.sbatch create example_configs/sweep-config.yaml   # ID is in the .out file
sbatch --array=1-8%2 --partition=<gpu-partition> --gres=gpu:1 \
  scripts/sweep.sbatch agent <entity/project/sweep-id> 1
```

The third argument is `RUN_COUNT`, the number of sequential trials per agent; the array
controls how many agents run in parallel. Authenticate with `wandb login` before
submitting, or export `WANDB_API_KEY` if local policy allows it.

## Evaluate a trained model

```bash
python scripts/validate.py \
  -config configs/my-run.yaml \
  -model /abs/path/model-best.h5 \
  -csv outputs/test_metrics.csv
```

Evaluates the main validation set plus every dataset under `additional_val_data_paths`,
prints a metric dictionary, and writes a two-column `metric,value` CSV when `-csv` is
given. It runs W&B in `disabled` mode, so nothing is logged. Use the **same config that
trained the model** — architecture and `model_max_seq_len` must match the checkpoint.

## Score new sequences

```bash
python scripts/get_activations.py \
  -model /abs/path/model-best.h5 \
  -in_files /abs/path/sequences.bed \
  -in_genomes /abs/path/hg38.fa \
  -out_file outputs/predictions.csv \
  --write_csv -score_column 0 --no_reverse_complement
```

- `-in_genomes` is omitted for FASTA input; `-in_files` and `-in_genomes` both accept
  several paths.
- `-score_column 0` is the output column for a regression model; for binary
  classification use `1` for the positive class.
- Without `--no_reverse_complement` the output holds separate rows for each forward
  sequence *and* its reverse complement — fine when averaging both strands on purpose,
  confusing when the user expected one row per interval. Choose deliberately.
- Drop `--write_csv` and pass `-layer_name <layer>` to dump intermediate-layer
  activations as a NumPy array instead of predictions — this is the route to embeddings.
- `--bayesian` enables the Bayesian-approximation (MC-dropout style) path.

## Interpret

See `interpretation.md`.
