"""
cluster × scene cosine-similarity mapping.
Usage: uv run python -X utf8 SMA/run_scene_map.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"
TOP_N    = 3
TOP_SHOW = 10  # clusters to display

# ── Load ──────────────────────────────────────────────────────────────
comments = pd.read_parquet(DATA_DIR / "comments_with_embedding.parquet")
clusters = pd.read_parquet(DATA_DIR / "best_clusters.parquet")
scenes   = pd.read_parquet(DATA_DIR / "scripts_with_embedding.parquet")

print(f"留言: {len(comments)}  場景: {len(scenes)}")
print(f"場景分布: {scenes.groupby('episode')['scene_number'].count().to_dict()}")

# ── Merge embeddings ──────────────────────────────────────────────────
df = clusters.merge(comments[["comment_id", "embedding"]], on="comment_id", how="left")
df_valid = df[df["cluster_label"] != -1]

# ── Cluster centroids ─────────────────────────────────────────────────
centroids = {}
for cid, grp in df_valid.groupby("cluster_label"):
    mat = np.stack(grp["embedding"].values)
    c   = mat.mean(axis=0)
    centroids[cid] = c / (np.linalg.norm(c) + 1e-9)

centroid_ids    = sorted(centroids.keys())
centroid_matrix = np.stack([centroids[c] for c in centroid_ids])

# ── Scene matrix ──────────────────────────────────────────────────────
scene_matrix = np.stack(scenes["embedding"].values)
scene_matrix = scene_matrix / (np.linalg.norm(scene_matrix, axis=1, keepdims=True) + 1e-9)

# ── Cosine similarity ─────────────────────────────────────────────────
sim = cosine_similarity(centroid_matrix, scene_matrix)
print(f"Similarity matrix: {sim.shape}  range [{sim.min():.3f}, {sim.max():.3f}]")

# ── Build results table ───────────────────────────────────────────────
cluster_stats = (
    df_valid.groupby("cluster_label")
    .agg(size=("comment_id", "count"), likes_total=("like_count", "sum"))
    .reset_index()
)

results = []
for i, cid in enumerate(centroid_ids):
    top_idx = sim[i].argsort()[-TOP_N:][::-1]
    for rank, sidx in enumerate(top_idx, 1):
        sr = scenes.iloc[sidx]
        results.append({
            "cluster_label":   cid,
            "scene_rank":      rank,
            "similarity":      float(sim[i, sidx]),
            "episode":         sr["episode"],
            "episode_num":     sr["episode_num"],
            "scene_number":    sr["scene_number"],
            "is_continuation": sr["is_continuation"],
            "scene_text":      sr["text"][:200],
        })

result_df = pd.DataFrame(results).merge(cluster_stats, on="cluster_label", how="left")

# ── Save CSV ──────────────────────────────────────────────────────────
out_path = DATA_DIR / "cluster_scene_mapping.csv"
result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"Saved → {out_path}  ({len(result_df)} rows)")

# ── Display top clusters ──────────────────────────────────────────────
top_clusters = (
    result_df[result_df["scene_rank"] == 1]
    .sort_values("likes_total", ascending=False)
    .head(TOP_SHOW)["cluster_label"]
    .tolist()
)

for cid in top_clusters:
    rows = result_df[result_df["cluster_label"] == cid].sort_values("scene_rank")
    stat = cluster_stats[cluster_stats["cluster_label"] == cid].iloc[0]
    top_cmts = (
        df_valid[df_valid["cluster_label"] == cid]
        .nlargest(2, "like_count")[["text_clean", "like_count"]]
    )
    print(f"\n{'='*65}")
    print(f"Cluster {cid}  size={stat['size']}  likes_total={stat['likes_total']}")
    for _, r in top_cmts.iterrows():
        print(f"  [{r['like_count']}讚] {str(r['text_clean'])[:70]}")
    print("  -- 最相似場景 --")
    for _, r in rows.iterrows():
        cont = "(续)" if r["is_continuation"] else ""
        print(f"  #{int(r['scene_rank'])} sim={r['similarity']:.3f}  "
              f"{r['episode']} 第{int(r['scene_number'])}幕{cont}")
        print(f"     {r['scene_text'][:90]}")

# ── Scene-level summary ───────────────────────────────────────────────
scene_hit = (
    result_df[result_df["scene_rank"] == 1]
    .groupby(["episode", "scene_number"])
    .agg(
        cluster_count=("cluster_label", "count"),
        total_likes=("likes_total", "sum"),
        avg_sim=("similarity", "mean"),
    )
    .reset_index()
    .sort_values("total_likes", ascending=False)
)

print("\n=== 各場景被引用情況（依讚數）===")
print(scene_hit.head(10).to_string(index=False))
