from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return pd.read_parquet(manifest_path)
