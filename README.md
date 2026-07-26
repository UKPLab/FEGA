# FEGA: Feature-Effect Geometry Analysis

FEGA studies the downstream geometry of sparse-autoencoder feature effects. For
one SAE feature, it compares the logit changes caused by removing that feature
across many contexts and asks a simple question: do those effects follow one
stable direction, occupy a low-dimensional structure, split into several modes,
or remain diffuse?

This repository accompanies **Sparse Autoencoders Encode Both Concepts and
Functions: The Downstream Geometry of Feature Effects**.

> **Paper:** public link pending before release.

The repository contains the FEGA implementation, experiment configurations,
paper-reproduction scripts, and retained upstream components needed by the
experiments. Generated datasets and results are not stored in Git.

## What FEGA produces

A FEGA run separates expensive model-facing work from downstream geometric
analysis:

| Phase | Purpose | Typical hardware |
| --- | --- | --- |
| `data_prep` | Resolve the selected contexts and SAE inputs. | CPU or GPU, depending on the source task |
| `compute_effect` | Compute the feature-removal logit effects. | GPU |
| `geometry_metrics` | Measure directional and residual geometry. | CPU |
| `vmf` | Fit directional mixtures to retained effect directions. | CPU |
| `stability` | Test the selected geometry family under resampling. | CPU |
| `geometry_reporting` | Write feature records, summaries, and the atlas. | CPU |

Each phase writes reusable artifacts beneath the configured run directory. You
can rerun downstream phases without loading the language model when the required
upstream artifacts already exist.

## Installation

The locked environment uses Python 3.10.19. Install
[uv](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/UKPLab/FEGA.git
cd FEGA
bash scripts/setup.sh
source scripts/activate.sh
python -m nltk.downloader brown
hf auth login
```

Gemma-2 and the SAE checkpoints require Hugging Face access. Authentication is
managed by your Hugging Face login or environment; `scripts/activate.sh` never
loads a machine-specific secret file.

`scripts/setup.sh` installs the frozen environment from `uv.lock`, installs the
FEGA spherecluster compatibility patch, and checks that patch. Do not regenerate
the lock when reproducing the paper environment.

## Run FEGA

The Python CLI accepts a pipeline YAML and either all phases or a comma-separated
subset:

```bash
python -m fega.cli run \
  --config fega/config/ravel/matryoshka/city_country_2pow16.yaml \
  --phases vmf,stability,geometry_reporting \
  --device cpu
```

Inspect the latest state for a configuration with:

```bash
python -m fega.cli status \
  --config fega/config/ravel/matryoshka/city_country_2pow16.yaml
```

The checked-in RAVEL YAML files are concrete examples. To use a different SAE or
attribute, edit or copy the matching file under `fega/config/ravel/` and update
its input, SAE, MDBM, output, and phase settings.

### Slurm launchers

The launchers expose the same CLI through environment variables. For example,
this submits the downstream CPU phases on the configured `yolo` partition:

```bash
FEGA_CONFIG=fega/config/ravel/matryoshka/city_country_2pow16.yaml \
FEGA_PHASES=vmf,stability,geometry_reporting \
bash scripts/slurm/sbatch_yolo.sh
```

Use `scripts/slurm/sbatch_cli_gpu.sh` for GPU-facing phases and
`scripts/slurm/sbatch_cli_cpu.sh` or `scripts/slurm/sbatch_yolo.sh` for
CPU-only downstream work. Change `FEGA_CONFIG` and `FEGA_PHASES`; the launcher
itself should not need editing.

## Reproduce the paper experiments

### Value-like RAVEL experiment

First generate the RAVEL and MDBM inputs on a GPU:

```bash
python external/custom_eval_instance.py --device cuda:0 --batch_size 128
```

The helper writes reusable inputs under:

```text
data/instance_eval_results/ravel/
data/artifacts/ravel/
data/mdbm/ravel/
```

Its configuration block selects the SAE repository, model, trainer, and
architecture. Adjust those values when generating inputs for ReLU, TopK, or
Matryoshka Batch TopK.

Then run the FEGA phases required by the selected YAML:

```bash
FEGA_CONFIG=fega/config/ravel/matryoshka/city_country_2pow16.yaml \
bash scripts/slurm/sbatch_yolo.sh
```

### Pointer-like and hybrid ICL experiments

Generate the LSC, WC, TT, and PrOntoQA datasets:

```bash
TASKS="lsc wc tt prontoqa" \
MODEL_NAME=gemma-2-2b \
TARGET_EXAMPLES=50000 \
NUM_FAMILIES=1000 \
SEED=0 \
BATCH_SIZE=32 \
DEVICE=cuda:0 \
EXTRA_ARGS="--family-pool-multiplier 10 --candidates-per-family-round 64 --max-candidates-per-family 100000" \
bash external/sae_bench/evals/icl_features/scripts/generate_datasets.sh
```

Validate the generated inputs, then run the three-SAE comparison:

```bash
MODEL_NAME=gemma-2-2b \
DATA_ROOT=data/icl_features \
bash external/sae_bench/evals/icl_features/scripts/validate_datasets.sh

RANDOM_TRIALS=20 \
DEVICE=cuda:0 \
bash external/sae_bench/evals/icl_features/scripts/run_saebench_gemma2b_width2pow16_three_saes.sh
```

For a small end-to-end check, use
`external/sae_bench/evals/icl_features/scripts/run_saebench_gemma2b_width2pow16_smoke_test.sh`.
The detailed experiment runbook is
[`external/sae_bench/evals/icl_features/README.md`](external/sae_bench/evals/icl_features/README.md).

## Numerical reproducibility

FEGA's reference vMF path is `dense_cpu`. It does not require CUDA once effect
artifacts exist. Run the same frozen environment and fixed thread topology when
comparing model selection or assignments across machines.

Floating-point values can change slightly with the processor, BLAS library,
threading, driver, or CUDA stack. The scientific portability target is therefore
the selected directional model and downstream geometry label, not byte-identical
JSON. Optional factorized CPU and GPU backends must pass the local policy check:

```bash
python scripts/validation/check_vmf_backends.py --backend cpu-factor
python scripts/validation/check_vmf_backends.py --backend gpu-factor --gpu-device cuda:0
```

If an optimized backend is not validated on the current machine, FEGA uses the
complete `dense_cpu` fit. Exact artifact reproduction additionally requires the
environment recorded in `fega/core/vmf/backend_policy.json`.

## Results and completion checks

A completed value-like run contains, beneath its configured run root:

```text
run_status.json
vmf/pre_softcap_logits/vmf_scores.json
stability/stability_scores.json
geometry_reporting/geometry_feature_records.json
geometry_reporting/figures/geometry_atlas.png
```

Use a result only after its requested phases are marked successful in
`run_status.json`. The pointer-like comparison records its final audit at:

```text
results/sae_geometry_gemma2b_65k/paper_artifact_audit.json
```

### Per-feature visualizations

Render the highest-support cached examples from every available geometry family
without loading the model or rerunning FEGA:

```bash
python -m fega.cli visualize \
  --run-dir results/fega/<dataset>/<sae-run>/<entity>_<attribute> \
  --top-n 5 \
  --dpi 300
```

Outputs are stored beside the other phases:

```text
visualizations/
  candidates.json
  candidates/<geometry-class>/rank_<NN>_f<feature-id>/
    sphere_ball.png
    sphere_surface.png
    projection_2d.png
    card.png
    metrics.json
```

The renderer reuses cached coordinates, selected residual dimensions, and vMF
assignments. Display-only centering, rotation, color, or guide lines are recorded
in `metrics.json` and do not change the family decision. A partial hex-color
palette can be passed with `--palette-json palette.json`.

## Repository layout

| Path | Contents |
| --- | --- |
| `fega/` | FEGA implementation and experiment configurations. |
| `scripts/` | Setup, local, validation, and Slurm entry points. |
| `tests/` | Unit, integration, and opt-in hardware tests. |
| `external/` | Retained upstream code and experiment wrappers. |
| `data/` | Generated datasets, caches, and intermediate inputs; ignored by Git. |
| `results/` | Generated experiment artifacts and figures; ignored by Git. |

## Citation

Please cite the accompanying paper when using FEGA:

> Phu Gia Hoang, Anwoy Chatterjee, Tanmoy Chakraborty, Iryna Gurevych, and
> Subhabrata Dutta. *Sparse Autoencoders Encode Both Concepts and Functions:
> The Downstream Geometry of Feature Effects*. 2026. Public link pending.

Machine-readable software and paper citation metadata are provided in
[`CITATION.cff`](CITATION.cff).

## Maintainer

**Phu Gia Hoang** — [hoanggiaphu26@gmail.com](mailto:hoanggiaphu26@gmail.com)

[Ubiquitous Knowledge Processing Lab](https://www.ukp.tu-darmstadt.de/) ·
[Technical University of Darmstadt](https://www.tu-darmstadt.de/)

## License and third-party software

FEGA-authored code is licensed under the [Apache License 2.0](LICENSE).
Retained or adapted third-party material remains under its own terms; see
[NOTICE](NOTICE) and the license files under `external/` and
`THIRD_PARTY_LICENSES/`.

`external/sae_bench` is retained for private research reproducibility, but its
upstream snapshot does not include a license. It is not covered by FEGA's
Apache-2.0 license and must not be publicly redistributed until its permission
status is resolved.

This repository contains experimental software and is published for the sole
purpose of giving additional background details on the respective publication.
