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

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_LOCAL_CONFIG_PATH = _CONFIG_DIR / "local_config.yaml"
_EXAMPLE_CONFIG_PATH = _CONFIG_DIR / "local_config.example.yaml"


def load_local_config() -> dict:
    """Load config/local_config.yaml and return its contents as a dict.

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
