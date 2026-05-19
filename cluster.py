"""
cluster.py — one-shot clustering with the best pipeline from autoresearch.

Pipeline (固定，從 autoresearch 找到的最佳組合):
    F2 prefilter (likes >= 2) → UMAP 50d cosine → HDBSCAN min_cluster=20 → P3 noise detector

Output:
    data/clusters.parquet  — comments with cluster_label, including noise (-1)

Usage:
    uv run final_project/cluster.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_comments, DATA_DIR

# ══════════════════════════════════════════════════════════════════════
# Best pipeline parameters (from autoresearch v2 / commit cfdfcd4)
# ══════════════════════════════════════════════════════════════════════
F_MIN_LIKES        = 2          # prefilter: drop very cold comments
UMAP_N_COMPONENTS  = 50
UMAP_N_NEIGHBORS   = 15
UMAP_MIN_DIST      = 0.0
MIN_CLUSTER_SIZE   = 20
MIN_SAMPLES        = 1
HDBSCAN_METHOD     = "eom"
INTERACTIVE_MIN_LIKES = 500     # P3: high-like noise comments threshold


def main():
    print("── Load ──")
    df = load_comments()
    print(f"  {len(df)} valid comments loaded")

    print(f"\n── Prefilter (likes >= {F_MIN_LIKES}) ──")
    mask = df["like_count"].values >= F_MIN_LIKES
    df = df[mask].reset_index(drop=True)
    X = np.stack(df["embedding"].values)
    print(f"  {len(df)} comments after filter, X shape: {X.shape}")

    print("\n── UMAP ──")
    import umap as umap_lib
    cache_path = DATA_DIR / "umap_50d_cache.npy"
    if cache_path.exists():
        print(f"  cache hit: {cache_path.name}")
        X_red = np.load(cache_path)
    else:
        print(f"  running UMAP n_components={UMAP_N_COMPONENTS} ...")
        reducer = umap_lib.UMAP(
            n_components=UMAP_N_COMPONENTS,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric="cosine",
            random_state=42,
            verbose=False,
        )
        X_red = reducer.fit_transform(X)
        np.save(cache_path, X_red)
        print(f"  cached → {cache_path.name}")
    print(f"  reduced shape: {X_red.shape}")

    print("\n── HDBSCAN ──")
    from sklearn.cluster import HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_method=HDBSCAN_METHOD,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X_red)
    df["cluster_label"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = (labels == -1).sum() / len(labels)
    print(f"  n_clusters={n_clusters}, noise_ratio={noise_ratio:.2%}")

    print("\n── P3 interactive meme detector ──")
    noise_mask = (df["cluster_label"] == -1) & (df["like_count"] >= INTERACTIVE_MIN_LIKES)
    df["is_interactive_candidate"] = noise_mask
    n_interactive = int(noise_mask.sum())
    print(f"  {n_interactive} noise comments with likes >= {INTERACTIVE_MIN_LIKES}")

    print("\n── Save ──")
    # Store both reduced coordinates and cluster label
    out = df[["comment_id", "text_clean", "like_count",
              "cluster_label", "is_interactive_candidate"]].copy()
    out["umap_x"] = X_red[:, 0]
    out["umap_y"] = X_red[:, 1]

    out_path = DATA_DIR / "clusters.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  Saved → {out_path}  ({len(out)} rows)")

    print(f"\nDone. {n_clusters} clusters + {(labels == -1).sum()} noise comments.")
    print(f"      {n_interactive} of the noise are interactive meme candidates.")


if __name__ == "__main__":
    main()
