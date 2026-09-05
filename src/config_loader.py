"""
Loader for this project's local, machine-specific configuration.

Why this exists
----------------
Every machine running this project's code has its own local filesystem layout
(e.g. a specific person's Windows path to a data directory). To keep those
real paths out of version control entirely, they live in
``config/local_config.yaml`` — a file that is gitignored and never committed.
Only ``config/local_config.example.yaml`` (a placeholder template) is tracked
by git. Anyone setting up the project copies the example file, fills in their
own real paths, and never touches git with that file again.

Scripts should import from this module to get local paths rather than ever
hardcoding a path directly.
"""

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _REPO_ROOT / "config"
_LOCAL_CONFIG_PATH = _CONFIG_DIR / "local_config.yaml"
_EXAMPLE_CONFIG_PATH = _CONFIG_DIR / "local_config.example.yaml"


def load_local_config() -> dict:
    """Load config/local_config.yaml and return its raw contents as a dict.

    Values are returned exactly as parsed from YAML (e.g. "data_root" comes
    back as a plain relative string). Prefer load_resolved_config() when you
    need actual filesystem paths, since raw values here are not anchored to
    the repo root and only work if the calling script's current working
    directory happens to be the repo root.

    Raises:
        FileNotFoundError: if config/local_config.yaml does not exist, with a
            message explaining how to create it from the example template.
    """
    if not _LOCAL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing local config file: {_LOCAL_CONFIG_PATH}\n"
            f"Copy {_EXAMPLE_CONFIG_PATH.name} to {_LOCAL_CONFIG_PATH.name} "
            "in the config/ directory and fill in your real local paths."
        )

    with open(_LOCAL_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_resolved_config() -> dict:
    """Load local_config.yaml with every path resolved to an absolute Path.

    load_local_config() returns raw YAML values verbatim, so "data_root"
    comes back as a bare relative string (e.g. "datasets") with no
    anchoring — it only resolves correctly if the calling script's current
    working directory happens to be the repo root. This function fixes that
    by anchoring "data_root" to the repo root using the same pattern already
    used for _CONFIG_DIR above (Path(__file__).resolve().parent.parent), and
    then joining every other filename-like key in the config onto that
    resolved data root. Callers get ready-to-use absolute paths, independent
    of where a script is run from.

    Returns:
        dict with:
            - "data_root": absolute Path to the data root directory.
            - one "<name>_path" absolute Path entry for every other key in
              local_config.yaml, derived from its key name (e.g.
              "zarr_filename" -> "zarr_path", "sim_pressures_csv" ->
              "sim_pressures_csv_path").
    """
    raw = load_local_config()

    data_root = (_REPO_ROOT / raw["data_root"]).resolve()
    resolved = {"data_root": data_root}

    for key, filename in raw.items():
        if key == "data_root":
            continue
        path_key = key[: -len("_filename")] if key.endswith("_filename") else key
        resolved[f"{path_key}_path"] = data_root / filename

    return resolved
