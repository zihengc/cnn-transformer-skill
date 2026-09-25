# cnn-transformer-skill

A [Claude Code](https://claude.com/claude-code) skill that teaches Claude to drive the
[`cnn-transformer-pipeline`](https://github.com/zihengc/cnn-transformer-pipeline):
training, tuning, evaluating, and interpreting CNN and CNN+RoPE-Transformer models on
regulatory DNA sequences.

The pipeline itself is **not** vendored here. This repository holds only the skill —
instructions, reference documentation, and a config validator — so that the pipeline has
exactly one source of truth.

## What it covers

- Writing and fixing the pipeline's `desc`/`value` YAML configs (data paths, zero-based
  BED target columns, regression vs. classification, CNN and transformer architecture).
- Running the included HEK293 ATAC-seq demo as a smoke test.
- Training locally or through the portable Slurm entry points.
- W&B hyperparameter sweeps.
- Evaluating a checkpoint (`scripts/validate.py`) and scoring new sequences
  (`scripts/get_activations.py`).
- DeepSHAP attribution and TF-MoDISco-lite motif discovery.
- Mapping the pipeline's common error messages back to their causes.

## Install

Personal skill, available in every project:

```bash
git clone https://github.com/zihengc/cnn-transformer-skill.git
mkdir -p ~/.claude/skills
cp -r cnn-transformer-skill/cnn-transformer-pipeline ~/.claude/skills/
```

Or project-scoped, committed alongside a piece of work:

```bash
mkdir -p .claude/skills
cp -r cnn-transformer-skill/cnn-transformer-pipeline .claude/skills/
```

Claude loads the skill's name and description at startup and reads the rest when a task
matches — ask something like "train a CNN on my ATAC-seq peaks with this pipeline" and
it will pick it up.

## Layout

```text
cnn-transformer-pipeline/
├── SKILL.md                       # entry point: workflow selection and working habits
├── references/
│   ├── configuration.md           # every config field, and the traps in each
│   ├── workflows.md               # exact commands for train / sweep / validate / predict
│   ├── interpretation.md          # DeepSHAP and TF-MoDISco-lite
│   └── troubleshooting.md         # symptom → cause → fix
└── scripts/
    └── check_config.py            # config validator, PyYAML only
```

## The config validator, standalone

Useful on its own, with or without Claude. It needs only PyYAML, so it runs before the
heavy TensorFlow environment exists:

```bash
python cnn-transformer-pipeline/scripts/check_config.py my-config.yaml
```

Run it from the pipeline repository root so relative data paths resolve, or pass
`--repo-root`. It reports unreplaced path placeholders, missing data files, mismatched
`*_data_paths`/`*_targets` lengths, per-layer lists that are too short, regression
configs left on a classification loss, half-configured transformer heads, and the
`4e-4`-parses-as-a-string YAML trap. Exit code is 1 when there are errors.

## Related

- Pipeline: [zihengc/cnn-transformer-pipeline](https://github.com/zihengc/cnn-transformer-pipeline)
- Upstream lab pipeline: [pfenninglab/cnn_pipeline](https://github.com/pfenninglab/cnn_pipeline)
- Publication: Chen et al., *Context-dependent regulatory variants in Alzheimer's
  disease*, bioRxiv 2025.07.11.659973,
  [doi:10.1101/2025.07.11.659973](https://doi.org/10.1101/2025.07.11.659973)

## License

MIT License. See [`LICENSE`](LICENSE).
