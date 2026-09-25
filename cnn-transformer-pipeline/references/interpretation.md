# DeepSHAP and TF-MoDISco-lite

Two entry points: a config-driven one for models trained by this pipeline, and a
config-free one for pre-extracted sequence files.

## Config-driven (`scripts/explain.py`)

Set these in the config first:

```yaml
interp_model_path:   { value: /abs/path/model-best.h5 }
shap_bg_data_paths:  { value: [ {genome: /abs/hg38.fa, intervals: data/val.Neg.bed} ] }
shap_fg_data_paths:  { value: [ {genome: /abs/hg38.fa, intervals: data/val.pos.bed} ] }
shap_num_bg:         { value: 1000 }
shap_num_fg:         { value: 1000 }
shap_pos_value:      { value: 2 }
modisco_max_seqlets: { value: 100000 }
modisco_output:      { value: outputs/modisco_report/modisco_results.h5 }
modisco_report_dir:  { value: outputs/modisco_report }
modisco_motifs_database: { value: '' }   # optional MEME file for motif matching
```

(Written flat here for brevity — in the real config each key keeps its `desc`/`value`
block.)

```bash
mkdir -p outputs/modisco_report
python scripts/explain.py -config configs/my-run.yaml
```

Background is sampled from `shap_bg_data_paths`; foreground is the sequences from
`shap_fg_data_paths` whose signal exceeds `shap_pos_value`. If the threshold is above
anything in the foreground file, the foreground comes out empty and the run fails
downstream — check the target column's value range against `shap_pos_value` before
launching a long job.

Outputs land in `modisco_report_dir`:

- `onehot_encoded_sequences.npz` — foreground sequences, length-last
- `attributions_from_shap.npz` — matching DeepSHAP attributions
- `modisco_results.h5` — discovered motif patterns
- an HTML report plus supporting files (motif matches only when a MEME database was
  given)

The script creates the report directory, shells out to the `modisco` CLI, and
**chdir's into the report directory** to generate the report — so pass relative paths
for other things carefully, and treat the process's working directory as changed after
it runs.

### Cost

DeepSHAP is the slow part, roughly linear in `shap_num_bg × shap_num_fg` and in model
size. For a functional check, drop both to ~50 and `modisco_max_seqlets` to a few
thousand; say explicitly that the resulting motifs are not interpretable, only proof
that the plumbing works. For real analysis the 1000/1000 defaults are a reasonable
starting point, on a GPU.

## Config-free (`scripts/explain_mpra.py`)

For MPRA-style data where background and foreground are already separate FASTA files:

```bash
python scripts/explain_mpra.py \
  -bg_data_paths /abs/background.fa \
  -fg_data_paths /abs/foreground.fa \
  -interp_model_path /abs/model-best.h5 \
  -report_dir outputs/mpra_shap
```

Writes the compressed one-hot and attribution arrays to `-report_dir`. It does not run
MoDISco — run the `modisco motifs` CLI on those two `.npz` files if motif discovery is
wanted.

## Other interpretation tools

`scripts/get_activations.py -layer_name <layer>` (without `--write_csv`) saves
intermediate activations as a NumPy array — useful for embedding analyses.
`sketches/embedding_visualization.py` and `sketches/explain.py` are exploratory
prototypes, not maintained entry points; read them before relying on them.
