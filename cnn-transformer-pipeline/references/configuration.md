# Configuration reference

Every entry point (`train.py`, `scripts/validate.py`, `scripts/explain.py`, the sweep)
reads the same YAML. `utils.get_config` flattens it by taking `value` from each
top-level key, so the `desc` text is documentation only — but keep it, because W&B
displays it and the templates rely on it.

```yaml
key_name:
  desc: Human-readable explanation.
  value: <the actual setting>
```

## Contents

- [Data inputs](#data-inputs)
- [Targets](#targets)
- [Training](#training)
- [Optimization and learning rate](#optimization-and-learning-rate)
- [Architecture](#architecture)
- [Interpretation keys](#interpretation-keys)
- [Required keys](#required-keys)

## Data inputs

Three input shapes are accepted, mixed freely within one list:

```yaml
train_data_paths:
  value:
    - genome: /abs/path/hg38.fa           # BED / narrowPeak: needs both keys
      intervals: data/train.pos.bed
    - /abs/path/sequences.fa              # FASTA: a bare path
```

`.bed`, `.narrowPeak`, and `.narrowPeak.gz` all go through pybedtools; BED coordinates
are 0-based half-open, as usual. FASTA goes through Biopython. The same structure is
used by `val_data_paths`, `shap_bg_data_paths`, and `shap_fg_data_paths`.

`additional_val_data_paths` is **one level deeper** — a list of datasets, each of which
is itself a list of path entries. Each inner dataset is evaluated separately after
validation, which is how the demo reports held-out test metrics during training:

```yaml
additional_val_data_paths:
  value:
    - - genome: /abs/path/hg38.fa
        intervals: data/test.withNeg.bed
```

Sequences are padded to a shared length. `model_max_seq_len` sets a **minimum** padded
length, not a truncation limit — a longer interval in the input raises the actual model
input length. `train.py` exports it as `CNN_PIPELINE_MODEL_MAX_SEQ_LEN`, and
`scripts/validate.py` re-exports it so a fresh process rebuilds the same input shape.
If validation crashes on shape mismatch, this key is the first thing to check.

## Targets

Each `*_data_paths` entry needs exactly one matching `*_targets` entry, in order. Two
forms:

```yaml
train_targets:
  value:
    - column: 6        # read the 7th BED column — the numbering is ZERO-BASED
    - 0                # constant target for every sequence in this file
```

`targets_are_classes` decides everything downstream:

- `false` — regression on continuous values. Pair with `mean_squared_error`,
  `mean_absolute_error`, `mean_absolute_percentage_error`, or `huber`, and monitor
  `val_mean_squared_error`. `class_weight` must stay `none`; the pipeline raises if a
  weighting scheme is set on a regression run.
- `true` — classification. Pair with `sparse_categorical_crossentropy`. Class labels are
  mapped to indices consistently across datasets (`utils.validate_datasets` rejects
  inconsistent mappings). `metric_pos_label` selects the positive class for binary
  metrics, and `class_weight` may be `reciprocal` (inverse-frequency) or `proportional`
  (weight = fraction of data *not* in that class).

A common layout for accessibility data: one positives file plus several negative
partitions, all reading the same signal column, where negatives carry zero signal.

## Training

| Key | Notes |
| --- | --- |
| `project` / `entity` | W&B project and account. `entity: null` uses the default. |
| `batch_size` | 512 in the demo; lower it first when the GPU runs out of memory. |
| `num_epochs` | Set to 1 for a smoke test. |
| `model_checkpoint` | `none`, or a `.h5` path to resume from. |
| `early_stopping_callbacks` | List of `tf.keras.callbacks.EarlyStopping` kwargs. `monitor` must name a metric the run actually produces. |
| `use_exact_val_metrics` | `true` materializes validation as an array instead of an endless generator. |
| `use_reverse_complement` | Adds reverse-complement sequences — doubles effective training data, and doubles the rows produced by `get_activations.py`. |

## Optimization and learning rate

`optimizer` is `adam` or `sgd`, with `optimizer_args` passed through.

`lr_schedule: exponential` uses `lr_init` and `lr_exp_decay_per_epoch` — the
straightforward choice, and what the demo and sweep use.

`lr_schedule: cyclic` uses `lr_init`, `lr_max`, `lr_cyc_num_cycles`, and
`lr_cyc_scale_fn` (`triangular` / `triangular2`), then fine-tunes for
`clr_tail_epochs` with a linear decay to `lr_init/10`. `momentum_schedule: cyclic`
(with `momentum_base` / `momentum_max`) is meaningful for SGD. To pick `lr_max`, run the
range test in `clr_rangetest.py`.

Regularization: `l2_reg_conv`, `l2_reg_dense`, `l2_reg_final`, `dropout_rate_conv`,
`dropout_rate_dense`. The conv/dense ones accept either a scalar applied to all layers
or a list with at least one value per layer.

## Architecture

CNN body — `num_conv_layers`, `conv_filters`, `conv_width`, `conv_stride`,
`max_pool_size`, `max_pool_stride`. Any of the per-layer parameters may be a list; if it
is, it needs at least `num_conv_layers` entries, otherwise
`utils.check_layerwise_params` raises. Same rule for `dense_filters` against
`num_dense_layers`.

Transformer head — set `use_transformer: true` **and** `num_transformer_layers` ≥ 1.
Setting one without the other is the usual way to accidentally train a CNN-only model
while believing otherwise. Then `transformer_d_model` (hidden size fed in from the CNN),
`transformer_num_heads`, `transformer_ff_dim`, `transformer_dropout`. Attention is
RoPE-based multi-head self-attention with padding-aware masking and pooling, so variable
interval lengths are handled without the padding leaking into attention.

Start from `config-base-cnn_transformer_template.yaml` rather than adding transformer
keys to a CNN config, so no key is missed.

## Interpretation keys

`shap_bg_data_paths` / `shap_fg_data_paths` (+ their `_targets`), `interp_model_path`,
`shap_num_bg`, `shap_num_fg`, `shap_pos_value`, `modisco_max_seqlets`,
`modisco_output`, `modisco_report_dir`, `modisco_motifs_database`. See
`interpretation.md`.

## Required keys

`utils.validate_config` asserts presence and type for: `project`, `train_data_paths`,
`train_targets`, `val_data_paths`, `val_targets`, `batch_size`, `num_epochs`,
`metric_pos_label`, `optimizer`, `lr_schedule`, `lr_cyc_scale_fn`, `lr_init`, `lr_max`,
`l2_reg_conv`, `l2_reg_dense`, `l2_reg_final`, `lr_exp_decay_per_epoch`,
`lr_cyc_num_cycles`, `dropout_rate_conv`, `dropout_rate_dense`, `num_conv_layers`,
`conv_filters`, `conv_width`, `conv_stride`, `max_pool_size`, `max_pool_stride`,
`num_dense_layers`, `dense_filters`, `shap_num_bg`, `shap_num_fg`.

The LR keys are required even when the other schedule is selected, so keep the unused
ones at their template values instead of deleting them. Note the types: `lr_init` and
the `l2_*`/`dropout_*` keys must parse as floats — write `4.e-4`, not `4e-4`, which
YAML 1.1 parses as a string.
