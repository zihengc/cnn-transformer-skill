# Troubleshooting

The pipeline's failures are mostly config failures wearing a TensorFlow costume. Match
the symptom, then re-run `python scripts/check_config.py <config>` — it catches most of
the table below before a job starts.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Bare `AssertionError` in `utils.validate_config` | A required key is missing, or has the wrong type | Compare against the required-key list in `configuration.md`; keep unused LR keys rather than deleting them |
| `Invalid type in config! key lr_init, expected [float], got <class 'str'>` | YAML 1.1 needs a dot in exponent notation | Write `4.e-4`, not `4e-4` |
| `assert len(paths) == len(targets)` | A data path was added without its target entry | Keep `*_data_paths` and `*_targets` the same length and order |
| `Not enough layer-wise params for parameter conv_filters` | A per-layer list is shorter than `num_conv_layers` | Extend the list, or use a scalar to apply one value to all layers |
| `FileNotFoundError` on a `/absolute/path/to/hg38.fa` | The template placeholder was never replaced | Replace every occurrence — the demo config has several |
| Relative `demo_data/...` paths not found | Running from somewhere other than the repo root | `cd` to the repository root |
| `Empty interval in BED file` / `No intervals in BED file` | Malformed, filtered-to-empty, or wrong-assembly BED | Inspect the file; confirm the chromosome names match the FASTA |
| Model input shape mismatch when validating | `model_max_seq_len` differs from training, or a longer interval raised the padded length | Validate with the config that trained the model |
| `class_weight` error on a regression run | A weighting scheme set while `targets_are_classes: false` | Set `class_weight: none` |
| Early stopping never triggers, or `monitor` warning | `monitor` names a metric this run does not emit (e.g. an MSE monitor on a classification run) | Match the monitor to the task: `val_mean_squared_error` for regression |
| Inconsistent `class_to_idx_mapping` | Classification datasets whose label sets differ | Make every dataset use the same label vocabulary |
| Sweep trains the demo instead of your model | `sweep-config.yaml` pins the demo config in its `command:` block | Edit the `-config` path in the sweep file |
| `wandb` hangs or errors with no network | Online mode by default | `-wandb-mode offline` (or `disabled`), and set `WANDB_DIR` to a writable path |
| Slurm job: "Conda/Mamba is unavailable in this batch job" | The cluster exposes conda through modules only | Load the module before `sbatch`, or export `CONDA_EXE` |
| Environment fails to solve | The lock targets linux-64 + TF 2.7 + Python 3.9 | Use a 64-bit Linux host; macOS/Windows/ARM are not supported |
| OOM during training | Batch too large for the GPU | Lower `batch_size`, then `conv_filters` / `transformer_d_model` |
| DeepSHAP runs for hours | `shap_num_bg × shap_num_fg` is large | Reduce both for a check; use a GPU for real runs |
| Empty SHAP foreground | `shap_pos_value` exceeds every signal value in the foreground file | Check the target column's range and lower the threshold |
| Transformer config trains a plain CNN | `use_transformer: true` without `num_transformer_layers`, or vice versa | Set both; prefer the transformer template |
| Twice as many prediction rows as intervals | Reverse-complement predictions included | Pass `--no_reverse_complement` to `get_activations.py` |
| Checkpoints "missing" after training | They are in the W&B run directory, not the repo root | Read the path W&B prints at init |
