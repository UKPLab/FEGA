**SAE Geometry ICL Pipeline**

This directory contains the reproducible pipeline used to isolate and analyze
pointer-like SAE features for the ICL tasks in the SAE Geometry paper. The
pipeline supports Literal Sequence Copying (LSC), Word Content (WC), Token
Translation (TT), and PrOntoQA, and it can be paired with the existing RAVEL
pipeline for value-like features.

The core workflow is:

1. Generate 50,000 model-correct examples per ICL task.
2. Validate the shared dataset schema and prompt-family balance.
3. Discover pointer-like SAE features using recurrence thresholds.
4. Run causal ablations with matched random controls.
5. Run FEGA geometry classification for the selected features.
6. Generate cross-task IoU matrices, paper plots, and LaTeX tables.

All generated JSON files are written with two-space indentation. The ICL
datasets share one flat schema and retain the original PrOntoQA-style fields:
`prompt`, `answer`, `x`, `y`, `entity`, `source_concept`, `target_concept`,
`support_example_index`, `lookup_rule`, and `induction_prefix`, with additional
task, family, token, and model-filter provenance.

**Prompt Families**

Prompt-family balance is part of the scientific design. By default, each task
contains 1,000 prompt families and 50 retained examples per family.

- `lsc`: one family fixes the repeated prefix/template; query tokens and random
  gap tokens vary.
- `wc`: one family fixes the feature-to-label mapping and demonstrations; query
  class and distractors vary.
- `tt`: one family fixes the language direction and in-context examples;
  held-out translation queries vary.
- `prontoqa`: one family fixes the support-rule context; query entity and reused
  rule vary.

Candidate generation is deterministic under `(seed, task, family_index)`.
Duplicate prompts within a family are rejected. Every retained row must have
the correct argmax first answer token under the selected Gemma model.

**Environment Setup**

Start from the environment that can already run the older PrOntoQA discovery
code. Then install the additional dependencies used by the geometry and table
pipeline:

```bash
python -m pip install "scipy>=1.11" "nltk>=3.8" \
  "umap-learn>=0.5.6,<0.6" "spherecluster>=0.1.7"
python -m pip install -e . --no-deps
python -m nltk.downloader brown
python scripts/bootstrap/install_vmf_spherecluster_patch.py
python scripts/bootstrap/install_vmf_spherecluster_patch.py --check
hf auth whoami
```

Gemma access must be available through Hugging Face. Supported model aliases in
this pipeline are `gemma-2-2b` and `gemma-2-9b`.

**Step 1: Generate ICL Datasets**

Generate the four 50k datasets for Gemma-2-2B:

```bash
MODEL_NAME=gemma-2-2b \
DEVICE=cuda:0 \
bash external/sae_bench/evals/icl_features/scripts/generate_datasets.sh
```

Useful controls:

```bash
TASKS="lsc wc tt prontoqa" \
TARGET_EXAMPLES=50000 \
NUM_FAMILIES=1000 \
SEED=0 \
BATCH_SIZE=32 \
EXTRA_ARGS="--family-pool-multiplier 10 --candidates-per-family-round 64 --max-candidates-per-family 100000" \
bash external/sae_bench/evals/icl_features/scripts/generate_datasets.sh
```

Outputs are written to:

```text
data/icl_features/<model-name>/<task>.json
```

Generation fails instead of silently underfilling if a task cannot meet the
requested family quotas.

**Step 2: Validate Datasets**

Run the structural validator before any full experiment:

```bash
MODEL_NAME=gemma-2-2b \
DATA_ROOT=data/icl_features \
bash external/sae_bench/evals/icl_features/scripts/validate_datasets.sh
```

This checks that all requested tasks have exactly 50,000 examples, 1,000
balanced prompt families, the shared schema, unique prompts within families,
Gemma provenance, and agreement between stored target and predicted token IDs.

For an independent GPU forward-pass recheck of all examples:

```bash
MODEL_NAME=gemma-2-2b \
DATA_ROOT=data/icl_features \
RECHECK_MODEL_CORRECT=1 \
DEVICE=cuda:0 \
bash external/sae_bench/evals/icl_features/scripts/validate_datasets.sh
```

**Step 3: Discover Pointer-Like Features**

Run discovery for one task and one or more SAEs:

```bash
TASK=lsc \
MODEL_NAME=gemma-2-2b \
DATASET_PATH=data/icl_features/gemma-2-2b/lsc.json \
OUTPUT_DIR=data/induction_feature_outputs/sae_geometry_gemma2b_65k/gemma-2-2b/lsc \
REPO_ID=canrager/saebench_gemma-2-2b_width-2pow16_date-0107 \
SAE_LOCATIONS="gemma-2-2b_standard_new_width-2pow16_date-0107/resid_post_layer_12/trainer_2" \
DEVICE=cuda:0 \
bash external/sae_bench/evals/icl_features/scripts/discover_pointer_features.sh
```

Default scientific thresholds:

- `ACTIVATION_THRESHOLD=0.0`
- `MIN_EXAMPLE_FRACTION=0.9`
- `MIN_QUERY_FRACTION=0.9`
- `MIN_FAMILY_FRACTION=0.9`

The threshold set contains features active in at least 90% of all examples and
active in at least 90% of queries for at least 90% of prompt families. The
strict set contains only features active on every analyzed example. Discovery
outputs include `summary.json`, per-SAE feature-set JSON files, feature CSVs,
and prevalence plots.

**Step 4: Run The Generic ICL Pipeline**

For a custom group of SAEs, use the generic stage runner:

```bash
MODEL_NAME=gemma-2-2b \
REPO_ID=canrager/saebench_gemma-2-2b_width-2pow16_date-0107 \
SAE_LOCATIONS="gemma-2-2b_standard_new_width-2pow16_date-0107/resid_post_layer_12/trainer_2 gemma-2-2b_top_k_width-2pow16_date-0107/resid_post_layer_12/trainer_2" \
TASKS="lsc wc tt prontoqa" \
DATA_ROOT=data/icl_features \
DISCOVERY_ROOT=data/induction_feature_outputs/sae_geometry_gemma2b_65k \
DOWNLOAD_SAES_DIR=data/downloaded_saes \
RESULT_ROOT=results/sae_geometry_gemma2b_65k \
STAGES="discovery ablation iou geometry plots aggregate" \
DEVICE=cuda:0 \
bash external/sae_bench/evals/icl_features/scripts/run_icl_pipeline.sh
```

Available stages are `discovery`, `ablation`, `iou`, `geometry`, `plots`, and
`aggregate`. The runner is resumable by default; set `RESUME=0` to recompute
stage outputs where the underlying stage supports overwriting.

Discovery datasets, feature summaries, CSVs, and plots are stored under
`DISCOVERY_ROOT`. Ablations, FEGA outputs, tables, and paper plots remain under
`RESULT_ROOT`.

Important controls:

```bash
TARGET_EXAMPLES=50000
MIN_EXAMPLE_FRACTION=0.9
MIN_QUERY_FRACTION=0.9
MIN_FAMILY_FRACTION=0.9
ABLATION_FEATURE_SETS="threshold"
ABLATION_POSITION=final
RANDOM_TRIALS=20
RANDOM_MATCH_POOL_SIZE=100
GEOMETRY_FEATURE_SET=candidate
PLOT_EMBEDDING=auto
DOWNLOAD_SAES_DIR=data/downloaded_saes
```

`RANDOM_TRIALS=20` is a practical minimum for the matched random-control
experiment. Use more trials for final publication runs when compute permits.

**Step 5: Run The SAEBench Three-SAE Comparison**

The release launcher for the matched Gemma-2-2B, layer-12, width `2^16`
comparison is:

```bash
bash external/sae_bench/evals/icl_features/scripts/run_saebench_gemma2b_width2pow16_three_saes.sh
```

This script fixes the three SAEBench checkpoints:

- ReLU: `gemma-2-2b_standard_new_width-2pow16_date-0107/resid_post_layer_12/trainer_2`
- TopK: `gemma-2-2b_top_k_width-2pow16_date-0107/resid_post_layer_12/trainer_2`
- Matryoshka Batch TopK: `gemma-2-2b_matryoshka_batch_top_k_width-2pow16_date-0107/resid_post_layer_12/trainer_2`

It validates the datasets, checks that each checkpoint has the expected trainer
class, performs a real model/SAE preflight, runs the requested stages, and
audits the expected paper artifacts at the end.

Example tmux launch:

```bash
mkdir -p data/logs
tmux new -s sae-geometry-release
RANDOM_TRIALS=20 \
DEVICE=cuda:0 \
bash external/sae_bench/evals/icl_features/scripts/run_saebench_gemma2b_width2pow16_three_saes.sh \
  2>&1 | tee "data/logs/sae_geometry_release_$(date +%Y%m%d_%H%M%S).log"
```

Detach with `Ctrl-b d` and reattach with:

```bash
tmux attach -t sae-geometry-release
```

For partial runs, set `STAGES`. For example:

```bash
STAGES="ablation iou aggregate" \
bash external/sae_bench/evals/icl_features/scripts/run_saebench_gemma2b_width2pow16_three_saes.sh
```

**Step 6: Run A Smoke Test**

The smoke test generates tiny datasets, loads the model and all three SAEs,
runs discovery, causal checks, FEGA geometry, plots, aggregation, and required
artifact checks:

```bash
bash external/sae_bench/evals/icl_features/scripts/run_saebench_gemma2b_width2pow16_smoke_test.sh
```

Smoke outputs are isolated under:

```text
data/icl_features_smoke/
data/induction_feature_outputs/sae_geometry_smoke/
results/sae_geometry_smoke/
```

Smoke-mode geometry may use a small fallback feature set if the tiny sample has
no feature satisfying the full 90/90/90 thresholds. That fallback is marked in
the generated FEGA config and is not used by the full experiment launcher.

**Step 7: Run Gemma Scope Width Sweeps**

For Gemma Scope residual SAEs, use:

```bash
MODEL_NAME=gemma-2-2b \
WIDTHS="65k 131k" \
DEVICE=cuda:0 \
bash external/sae_bench/evals/icl_features/scripts/run_gemmascope_width_sweep.sh
```

For Gemma-2-9B:

```bash
MODEL_NAME=gemma-2-9b \
WIDTHS="65k 131k" \
DEVICE=cuda:0 \
bash external/sae_bench/evals/icl_features/scripts/run_gemmascope_width_sweep.sh
```

Defaults:

- Gemma-2-2B uses layer 12.
- Gemma-2-9B uses layer 20.
- Default L0s are `65k/l0-72` and `131k/l0-67` for Gemma-2-2B.
- Default L0s are `65k/l0-55` and `131k/l0-62` for Gemma-2-9B.

Override these with `GEMMASCOPE_LAYER`, `GEMMASCOPE_65K_L0`, and
`GEMMASCOPE_131K_L0`.

**Step 8: Run RAVEL Value-Feature Discovery And Geometry**

RAVEL is handled by a separate wrapper so the ICL logic remains unchanged. The
wrapper calls the existing RAVEL MDBM training/evaluation code, runs FEGA on the
selected value-like features, and writes RAVEL discovery summaries.

Reusable RAVEL inputs are stored under `data/instance_eval_results/ravel`,
`data/mdbm/ravel`, and `data/artifacts/ravel`; custom SAE downloads use
`data/downloaded_saes`. `--result-root` contains only FEGA and paper outputs.
Pass `--data-root` only when the shared data directory must live elsewhere.

For the SAEBench Gemma-2-2B width `2^16` three-SAE comparison:

```bash
python -m sae_bench.evals.icl_features.ravel_value_geometry \
  --model-name gemma-2-2b \
  --device cuda:0 \
  --result-root results/sae_geometry_gemma2b_65k \
  --source custom_repo \
  --repo-id canrager/saebench_gemma-2-2b_width-2pow16_date-0107 \
  --sae-location gemma-2-2b_standard_new_width-2pow16_date-0107/resid_post_layer_12/trainer_2 \
  --sae-location gemma-2-2b_top_k_width-2pow16_date-0107/resid_post_layer_12/trainer_2 \
  --sae-location gemma-2-2b_matryoshka_batch_top_k_width-2pow16_date-0107/resid_post_layer_12/trainer_2 \
  --stages ravel fega summarize \
  --entity city \
  --attribute Country \
  --ravel-attributes Country Continent Language \
  --kept-threshold 8 \
  --kept-op ge \
  --top-n-entities 500 \
  --top-n-templates 90 \
  --num-pairs-per-attribute 5000 \
  --train-test-split 0.7 \
  --llm-batch-size 32 \
  --random-seed 42
```

Use `--kept-op gt` only if reproducing a prior result that explicitly filtered
RAVEL features with `kept > 8`; the current table scripts default to
`kept >= 8`.

For Gemma Scope, use `--source sae_lens`, `--sae-release`, and `--sae-id`:

```bash
python -m sae_bench.evals.icl_features.ravel_value_geometry \
  --model-name gemma-2-2b \
  --device cuda:0 \
  --result-root results/sae_geometry_gemmascope/gemma-2-2b_width-65k_l0-72 \
  --source sae_lens \
  --sae-release gemma-scope-2b-pt-res \
  --sae-id layer_12/width_65k/average_l0_72 \
  --stages ravel fega summarize \
  --entity city \
  --attribute Country \
  --ravel-attributes Country Continent Language \
  --kept-threshold 8 \
  --kept-op ge \
  --top-n-entities 500 \
  --top-n-templates 90 \
  --num-pairs-per-attribute 5000 \
  --train-test-split 0.7 \
  --llm-batch-size 32 \
  --random-seed 42
```

To rerun only FEGA and summarization from existing RAVEL/MDBM artifacts, omit
the `ravel` stage:

```bash
--stages fega summarize
```

To separate the model-dependent GPU work from the dense CPU work, reuse all
arguments from the three-SAE command and change only the stage options:

```text
# Generate or reuse the RAVEL JSON and MDBM checkpoints.
--device cuda:0 --stages ravel

# Generate the per-SAE FEGA config and run the model-dependent phases.
--device cuda:0 --stages fega \
  --fega-phases data_prep,compute_effect,geometry_metrics

# Reuse those outputs for dense vMF, stability, reporting, and summarization.
--device cpu --stages fega summarize \
  --fega-phases vmf,stability,geometry_reporting
```

The wrapper keeps the base config's fixed dense worker settings. `--device`
controls model and effect materialization; it does not change the vMF backend
from `dense_cpu`.

The legacy standalone command is:

```bash
bash external/custom_eval_instance.sh
```

It runs only the Matryoshka trainer-2 SAE, forces RAVEL to rerun, and is not the
three-SAE reproduction command. Prefer `ravel_value_geometry.py --stages ravel`
when all three explicit SAEs and normal RAVEL reuse are required.

Each wrapper FEGA stage writes:

```text
<result-root>/<model>/<sae-uid>/ravel/city_Country/fega_config.yaml
```

That generated config can be submitted through the custom Slurm launchers:

```bash
FEGA_CONFIG="<path-to-generated-fega_config.yaml>" \
FEGA_PHASES=data_prep,compute_effect,geometry_metrics \
bash scripts/slurm/sbatch_cli_gpu.sh

FEGA_CONFIG="<path-to-generated-fega_config.yaml>" \
FEGA_PHASES=vmf,stability,geometry_reporting \
bash scripts/slurm/sbatch_cli_cpu.sh
```

The GPU launcher reserves a GPU for the model-dependent phases. Dense vMF and
stability remain CPU computations. Both Slurm launchers derive worker counts
from their allocations, unlike the fixed-worker one-shot path.

The selected RAVEL feature IDs are written to:

```text
<result-root>/<model>/<sae-uid>/ravel/city_Country/value_features/feature_ids.json
```

**Step 9: Generate Plots**

Each individual SAE already receives cross-task ICL geometry plots during the
`plots` stage. To make the three-SAE paper figure from saved plot data:

```bash
python -m sae_bench.evals.icl_features.geometry_sae_grid_plots \
  --sae-plot "ReLU=results/sae_geometry_gemma2b_65k/gemma-2-2b/<relu-sae-uid>/cross_task/geometry_plots/geometry_plot_data.csv" \
  --sae-plot "TopK=results/sae_geometry_gemma2b_65k/gemma-2-2b/<topk-sae-uid>/cross_task/geometry_plots/geometry_plot_data.csv" \
  --sae-plot "Matryoshka Batch TopK=results/sae_geometry_gemma2b_65k/gemma-2-2b/<matryoshka-sae-uid>/cross_task/geometry_plots/geometry_plot_data.csv" \
  --output-dir results/sae_geometry_gemma2b_65k/gemma-2-2b/cross_sae/geometry_plots \
  --basename geometry_three_saes
```

To make the matching RAVEL geometry figure:

```bash
python -m sae_bench.evals.icl_features.ravel_geometry_sae_grid_plots \
  --sae-map "ReLU=results/sae_geometry_gemma2b_65k/gemma-2-2b/<relu-sae-uid>/ravel/city_Country/fega/ravel/<ravel-json>/city_Country/geometry_reporting/geometry_map_data.json" \
  --sae-map "TopK=results/sae_geometry_gemma2b_65k/gemma-2-2b/<topk-sae-uid>/ravel/city_Country/fega/ravel/<ravel-json>/city_Country/geometry_reporting/geometry_map_data.json" \
  --sae-map "Matryoshka Batch TopK=results/sae_geometry_gemma2b_65k/gemma-2-2b/<matryoshka-sae-uid>/ravel/city_Country/fega/ravel/<ravel-json>/city_Country/geometry_reporting/geometry_map_data.json" \
  --output-dir results/sae_geometry_gemma2b_65k/gemma-2-2b/cross_sae/ravel_geometry_plots \
  --basename ravel_geometry_three_saes \
  --model-name gemma-2-2b \
  --sae-width 2pow16
```

To plot RAVEL-inclusive five-task IoU heatmaps:

```bash
python -m sae_bench.evals.icl_features.ravel_iou_heatmap \
  --result-root results/sae_geometry_gemma2b_65k \
  --model-name gemma-2-2b \
  --task-order lsc wc prontoqa tt ravel \
  --basename ravel_candidate_iou_three_saes
```

Plot scripts write both `.png` and `.pdf` files, plus data/metadata artifacts
where appropriate. File names include model/SAE tags to reduce ambiguity when
copying figures out of the result tree.

**Step 10: Generate LaTeX Tables**

Feature-count table:

```bash
python -m sae_bench.evals.icl_features.feature_count_latex_table \
  --result-root results/sae_geometry_gemma2b_65k \
  --discovery-root data/induction_feature_outputs/sae_geometry_gemma2b_65k \
  --model-name gemma-2-2b \
  --sae-width 2pow16 \
  --feature-set candidate \
  --output results/sae_geometry_gemma2b_65k/tables/feature_counts_latex.txt \
  --overwrite
```

Ablation table:

```bash
python -m sae_bench.evals.icl_features.ablation_latex_table \
  --result-root results/sae_geometry_gemma2b_65k \
  --model-name gemma-2-2b \
  --sae-width 2pow16 \
  --feature-set threshold \
  --output results/sae_geometry_gemma2b_65k/tables/ablation_latex.txt \
  --overwrite
```

Geometry-count table:

```bash
python -m sae_bench.evals.icl_features.geometry_counts_latex_table \
  --result-root results/sae_geometry_gemma2b_65k \
  --discovery-root data/induction_feature_outputs/sae_geometry_gemma2b_65k \
  --model-name gemma-2-2b \
  --sae-width 2pow16 \
  --geometry-feature-set candidate \
  --output results/sae_geometry_gemma2b_65k/tables/geometry_counts_latex.txt \
  --overwrite
```

The default task order used by the plotting and table scripts is:

```text
LSC, WC, PrOntoQA, TT, RAVEL
```

**Causal-Ablation Protocol**

The primary intervention zeroes all selected features together at the final
pre-answer position. The ablation runner decodes the modified SAE
representation and adds the original SAE reconstruction error back, isolating
the chosen latents without replacing the residual stream with a full imperfect
SAE reconstruction.

Accuracy outcomes are paired by example. Accuracy drops use a one-sided exact
McNemar test, with the two-sided value also stored. Random controls use
equally sized feature sets matched by activation prevalence and active
magnitude. Each random trial receives its own paired test, and the summary also
records the finite-sample corrected empirical probability that a matched-random
drop is at least as large as the selected-feature drop.

The ablation runner evaluates the baseline-correct subset by default and stores
the effective analyzed sample in the result JSON. This avoids mixing causal
effects with examples that the unmodified model did not solve.

**Data And Result Layout**

```text
data/induction_feature_outputs/sae_geometry_gemma2b_65k/
  gemma-2-2b/
    <task>/
      summary.json
      feature_metrics.csv
      per_sae/*_feature_set.json

results/sae_geometry_gemma2b_65k/
  gemma-2-2b/
    <sae_uid>/
      <task>/
        ablation/<threshold|strict>/
          ablation_summary.json
          selected_ablation_table.csv
          random_ablation_table.csv
          random_ablation_aggregate.csv
          condition_accuracies.csv
          selected_outcomes.csv.gz
          random_outcomes.csv.gz
        fega/
          .../geometry_reporting/
            geometry_feature_records.json
            geometry_feature_records.csv
            geometry_reporting_counts.csv
            geometry_map_data.json
            figures/geometry_atlas.png
      cross_task/
        iou/
          threshold_iou_matrix.csv
          strict_iou_matrix.csv
        geometry_plots/
          geometry_all_tasks.{png,pdf}
          geometry_task_panels.{png,pdf}
          separate/geometry_<task>.{png,pdf}
          geometry_plot_data.csv
          geometry_category_counts.csv
    cross_sae/
      geometry_plots/
      ravel_geometry_plots/
      iou_heatmaps/
  tables/
    feature_counts.csv
    ablation_results.csv
    random_control_results.csv
    geometry_category_counts.csv
    *_latex.txt
```

At the end of the fixed SAEBench three-SAE run,
`paper_artifact_audit.json` verifies the expected paper artifacts. For
intentional partial-stage runs, set `RUN_PAPER_AUDIT=0`.

**Release Notes**

- The 50,000-example task datasets must already exist before a full run unless
  the generation step is launched separately.
- FEGA geometry is independent of ablation results; both depend on the feature
  sets saved by discovery.
- RAVEL is a separate value-feature pipeline. ICL runs do not regenerate RAVEL
  artifacts unless `ravel_value_geometry.py` is called.
- The plotting scripts regenerate figures quickly from saved map/data files, so
  visual styling changes do not require rerunning model or FEGA computation.
