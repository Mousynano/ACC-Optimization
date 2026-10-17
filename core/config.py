"""Shared paths and the frozen experiment manifest for the public release."""

from functools import lru_cache
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT_DIR / "experiment_manifest_acc_cost_fje_v2_2.json"


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path
