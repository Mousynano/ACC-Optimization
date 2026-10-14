"""Shared experiment configuration loaded from a versioned manifest.

Framework V2.2-FJE is the default active research configuration.  Set ``ACC_MANIFEST`` to a
manifest filename/path to reproduce an older campaign without editing code.
"""

from functools import lru_cache
import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT_DIR / "experiment_manifest_acc_cost_fje_v2_2.json"


def _resolve_manifest_path() -> Path:
    override = os.getenv("ACC_MANIFEST")
    if not override:
        return DEFAULT_MANIFEST
    path = Path(override)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


MANIFEST_PATH = _resolve_manifest_path()


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_json_relative(path_value: str) -> dict:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path
