"""
Data loading utilities for final_project.
Simplified from SMA/prepare_sma.py — no autoresearch evaluation, no ground_truth required.
"""
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def load_comments() -> pd.DataFrame:
    """Load YouTube comments with 4096-dim Qwen3 embeddings."""
    df = pd.read_parquet(DATA_DIR / "comments_with_embedding.parquet")
    return df[df["keep"] == True].reset_index(drop=True)


def load_scenes() -> pd.DataFrame:
    """Load drama scenes (scene-level embedding, 27 scenes)."""
    return pd.read_parquet(DATA_DIR / "scripts_with_embedding.parquet")


def load_lines() -> pd.DataFrame:
    """Load drama lines (line-level embedding, all dialogue lines)."""
    path = DATA_DIR / "lines_with_embedding.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run embed_lines.py first."
        )
    return pd.read_parquet(path)


def load_ground_truth() -> dict:
    with open(ROOT / "ground_truth.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_clusters() -> pd.DataFrame:
    """Load cluster results produced by cluster.py."""
    path = DATA_DIR / "clusters.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run cluster.py first."
        )
    return pd.read_parquet(path)
