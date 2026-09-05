# Project With Shmuel — The Alpha Paper

This project measures **alpha = P_tot / P_th**, the ratio of total to thermal
pressure, across the local interstellar medium (ISM) within roughly 500 pc of
the Sun. The goal is to quantify how much non-thermal pressure support —
turbulence, magnetic fields, and cosmic rays — is needed, alongside thermal
pressure, to maintain vertical hydrostatic equilibrium in the local ISM.
Supervisor: **Prof. Shmuel Bialy**.

## Repository structure

- `config/` — Configuration templates and (locally, but not in git) the real
  machine-specific config file. See "Setup instructions" below.
- `src/` — Shared Python modules used across the project: conventions
  (colors, thresholds, plot style) and the config loader.
- `pipeline/` — Analysis and figure-generation scripts (compute/design pairs;
  see "Pipeline pattern" below). Currently empty.
- `figures/` — Generated figure output. Currently empty.
- `cache/` — Cached intermediate compute results (`.npz` files), regenerated
  from source data as needed. Gitignored; currently empty.
- `docs/` — Project documentation. Currently empty.

## Setup instructions

1. Clone the repository:
   ```
   git clone https://github.com/MatanBerko/Project_With_Shmuel.git
   ```
2. Create the conda environment from `environment.yml`:
   ```
   conda env create -f environment.yml
   conda activate research311
   ```
3. Copy the config template and fill in your real, machine-specific paths:
   ```
   cp config/local_config.example.yaml config/local_config.yaml
   ```
   Then edit `config/local_config.yaml` with your actual local paths. This
   file is **gitignored** — it is specific to your machine and must never be
   committed.

## Pipeline pattern

Every figure in this project follows a two-script convention:

- **`compute_*.py`** — reads source data, performs the actual computation,
  and caches the results to `cache/*.npz`.
- **`design_*.py`** — loads the cached `.npz` output and produces the final
  plot.

`design_*.py` scripts should contain **zero data-masking or filtering
logic** — only presentation logic (styling, labels, layout). Any decision
about what data to include or exclude belongs in the corresponding
`compute_*.py` script.

## Status

This repository has just been scaffolded. No analysis pipeline, data
loading, or plotting code exists yet — only the directory structure,
conventions module, config-loading pattern, and environment definition
described above.
