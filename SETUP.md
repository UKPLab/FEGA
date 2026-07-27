# Setup

## 1. Requirements and Python environment

FEGA's frozen reproduction environment uses:

- Linux;
- Python 3.10.19;
- [`uv`](https://docs.astral.sh/uv/);
- Hugging Face access for Gemma and SAE checkpoints; and
- Slurm only when using the included cluster launchers.

From the repository root:

```bash
git clone https://github.com/UKPLab/FEGA.git
cd FEGA
bash scripts/setup.sh
source scripts/activate.sh
python -m nltk.downloader brown
hf auth login
```

`scripts/setup.sh` installs the versions in `uv.lock` and the local spherecluster
fix. Do not regenerate `uv.lock` when reproducing the paper. Source
`scripts/activate.sh` in each new shell.

## 2. Authentication and cache paths

Run `hf auth login` before a step that downloads a model or SAE.

By default, caches live beneath `${XDG_CACHE_HOME:-$HOME/.cache}/fega`. On a
cluster where compute nodes cannot write to `$HOME`, select a shared writable
location before activation:

```bash
export SAE_CACHE_ROOT="$PWD/data/cache"
source scripts/activate.sh
```

The activation script uses or sets the following cache variables:

| Variable | Purpose |
| --- | --- |
| `SAE_CACHE_ROOT` | Common FEGA cache root |
| `HF_HOME` | Hugging Face configuration and cache root |
| `HF_HUB_CACHE` | Hugging Face Hub downloads |
| `NLTK_DATA` | NLTK resources |
| `MPLCONFIGDIR` | Matplotlib cache and configuration |
| `XDG_CACHE_HOME` | General user cache root |

## 3. Configure an experiment

Checked-in pipeline configurations live under `fega/config/`:

```text
fega/config/
├── ravel/
│   ├── relu/
│   ├── topk/
│   └── matryoshka/
└── induction/
```

Copy or edit the closest example. The main fields are:

- `reference_json`: source evaluation artifact;
- `sae_repo_id`: Hugging Face SAE repository;
- `output_root`: FEGA result root;
- `download_saes_dir`: local SAE download directory;
- `mdbm_root` or `mdbm_weight_path`: MDBM inputs;
- `entity_attribute_selection`: evaluated entity and attribute;
- `device`: default execution device; and
- `phases`: enabled phases and their settings.

Paths in a YAML file are relative to that file. Keep research settings in the
YAML. Use command options only to select phases, a device, or CPU workers.

## 4. Run FEGA

### Direct CLI

Run every enabled phase:

```bash
python -m fega.cli run \
  --config fega/config/ravel/matryoshka/city_country_2pow16.yaml
```

Run selected downstream phases from cached inputs:

```bash
python -m fega.cli run \
  --config fega/config/ravel/matryoshka/city_country_2pow16.yaml \
  --phases vmf,stability,geometry_reporting \
  --device cpu \
  --resume
```

Inspect the latest recorded state:

```bash
python -m fega.cli status \
  --config fega/config/ravel/matryoshka/city_country_2pow16.yaml
```

The phase order is `data_prep`, `compute_effect`, `geometry_metrics`, `vmf`,
`stability`, and `geometry_reporting`. `--resume` skips phases already marked
successful in `run_status.json`.

### Slurm hardware split

For a RAVEL run, first create the RAVEL and MDBM files with
`external/custom_eval_instance.py`. Then submit the GPU and CPU jobs in order:

```bash
export FEGA_CONFIG=fega/config/ravel/matryoshka/city_country_2pow16.yaml
bash scripts/slurm/sbatch_cli_gpu.sh
# Submit the next command after the GPU phases complete successfully.
bash scripts/slurm/sbatch_yolo.sh
```

The RAVEL launchers accept `FEGA_CONFIG`, `FEGA_PHASES`, `FEGA_DEVICE`, and
`FEGA_NUMERICAL_THREADS`. The induction launcher uses `FEGA_INDUCTION_CONFIG`
and also accepts `FEGA_PHASES`, `FEGA_DEVICE`, `FEGA_NUMERICAL_THREADS`, and
`FEGA_RESUME`.

## 5. Script guide

### Environment and local execution

| Script | How to use it |
| --- | --- |
| `scripts/setup.sh` | Run once from a fresh clone to install and verify the frozen environment. |
| `scripts/activate.sh` | Source in each shell: `source scripts/activate.sh`. |
| `scripts/local/cli.sh` | Local example that runs all phases and logs to `data/logs/`. Its current config path is not checked in, so use the direct CLI or change `CONFIG_PATH`. |

### Slurm launchers

The RAVEL launchers write `slurm_<job-id>.out` and `slurm_<job-id>.err` beneath
`data/logs/`; the induction launcher uses the corresponding
`induction_<job-id>` prefix.

| Script | Current default |
| --- | --- |
| `scripts/slurm/sbatch_cli_gpu.sh` | A100 job for `data_prep,compute_effect,geometry_metrics`; TopK RAVEL config; `cuda:0` |
| `scripts/slurm/sbatch_cli_cpu.sh` | CPU job for `geometry_reporting`; ReLU RAVEL config; `cpu` |
| `scripts/slurm/sbatch_yolo.sh` | Large CPU job for `vmf,stability,geometry_reporting`; TopK RAVEL config; `cpu` |
| `scripts/slurm/sbatch_induction.sh` | A100 induction run; full configured pipeline unless `FEGA_PHASES` is set; resume enabled |

Override a launcher without editing it:

```bash
FEGA_CONFIG=fega/config/ravel/relu/city_country_2pow16.yaml \
FEGA_PHASES=vmf,stability,geometry_reporting \
FEGA_DEVICE=cpu \
FEGA_NUMERICAL_THREADS=16 \
bash scripts/slurm/sbatch_yolo.sh
```

`FEGA_NUMERICAL_THREADS` must divide the CPU count requested by the launcher.

### Bootstrap and validation utilities

| Script | Intended use |
| --- | --- |
| `scripts/bootstrap/install_vmf_spherecluster_patch.py` | Install or check the local spherecluster fix. `scripts/setup.sh` normally runs it. |
| `scripts/validation/check_vmf_backends.py` | Check whether a faster vMF backend is safe to use on this machine. |
| `scripts/validation/validate_fega_gram_logit_equivalence.py` | Check that the two effect calculations agree on real saved data. Not needed for setup. |

These checks are optional. They are not FEGA phases.

Optional optimized-backend checks are:

```bash
python scripts/validation/check_vmf_backends.py --backend cpu-factor
python scripts/validation/check_vmf_backends.py \
  --backend gpu-factor \
  --gpu-device cuda:0
```

## 6. Results, status, and cached visualizations

Each config sets its result folder through `output_root`. Before using a result,
check that every requested phase is successful in `run_status.json`.

```text
run_status.json
vmf/pre_softcap_logits/vmf_scores.json
stability/stability_scores.json
geometry_reporting/geometry_feature_records.json
geometry_reporting/figures/geometry_atlas.png
```

Generate per-feature visualizations from a completed cached run:

```bash
python -m fega.cli visualize \
  --run-dir results/fega/<dataset>/<sae-run>/<entity>_<attribute> \
  --top-n 5 \
  --dpi 300
```

This command does not load the model or rerun FEGA. Use
`--palette-json palette.json` to change colors with `#RRGGBB` values.

## 7. Data and models

- Gemma models and SAE checkpoints are downloaded from Hugging Face and are not
  redistributed here.
- RAVEL evaluation, MDBM, ICL-task, cache, and intermediate artifacts live under
  `data/` and are excluded from Git.
- FEGA experiment outputs live under `results/` and are excluded from Git.
- Retained upstream research components live under `external/`; their licenses
  and distribution status are recorded in [`NOTICE`](NOTICE).

## 8. Verification

After setup, check the installed numerical patch:

```bash
python scripts/bootstrap/install_vmf_spherecluster_patch.py --check
```

Then inspect a configured run without starting new work:

```bash
python -m fega.cli status \
  --config fega/config/ravel/matryoshka/city_country_2pow16.yaml
```

A full experiment is not a setup check. Run GPU checks on a GPU node and CPU
checks on a CPU node.

## 9. Troubleshooting

### The compute node cannot write to `$HOME`

Set a shared cache root before activation:

```bash
export SAE_CACHE_ROOT="$PWD/data/cache"
source scripts/activate.sh
```

### A model or SAE download is unauthorized

Run `hf auth login` in the calling environment and confirm that the relevant
model or SAE repository has been accepted or granted to the account.

### An optimized vMF backend is unavailable

Use `dense_cpu`. The faster CPU and GPU paths are optional.

### A downstream phase refuses to run

Inspect `run_status.json` and the corresponding Slurm error log. Confirm that the
required upstream phase artifacts exist and were produced for the same configured
experiment before rerunning with `--resume`.

## 10. Licensing

FEGA-authored code is released under the [Apache License 2.0](LICENSE).
Third-party components retain their own terms; see [`NOTICE`](NOTICE), the
license files under `external/`, and `THIRD_PARTY_LICENSES/`.

The retained `external/sae_bench` snapshot has no confirmed license and is not
covered by FEGA's Apache-2.0 license. It must not be publicly redistributed until
permission or an applicable license is confirmed.
