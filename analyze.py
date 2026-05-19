"""
analyze.py — 對 cluster.py 的結果做場景對應、關鍵字抽取、meme quality 評分。

Output:
    data/cluster_analysis.parquet  — 每群一筆，含 scene/keywords/meme_quality
    data/comment_scene_mapping.parquet — 每則留言對應到的最近場景

Usage:
    uv run final_project/analyze.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_comments, load_scenes, load_clusters, DATA_DIR

# 四個訊號的權重（autoresearch S3 配置）
W_ENGAGE = 0.40
W_KW     = 0.30
W_SCENE  = 0.20
W_VAR    = 0.10

NGRAM_RANGE  = (2, 4)
TOP_KEYWORDS = 8


def compute_scene_sims(comments_emb: np.ndarray, scene_emb: np.ndarray) -> np.ndarray:
    """每則留言對 27 個場景的 cosine similarity，shape (n_comments, 27)."""
    c = comments_emb / (np.linalg.norm(comments_emb, axis=1, keepdims=True) + 1e-9)
    s = scene_emb   / (np.linalg.norm(scene_emb,   axis=1, keepdims=True) + 1e-9)
    return c @ s.T


def c_tf_idf(valid_df: pd.DataFrame, cluster_ids: list) -> tuple[dict, dict]:
    """Class-level TF-IDF on char n-grams. Returns (keywords, distinctiveness)."""
    from sklearn.feature_extraction.text import CountVectorizer
    docs = [" ".join(valid_df[valid_df["cluster_label"] == cid]["text_clean"].dropna())
            for cid in cluster_ids]
    try:
        vec = CountVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE, min_df=2)
        tf_mat = vec.fit_transform(docs).toarray().astype(float)
    except ValueError:
        return {c: [] for c in cluster_ids}, {c: 0.0 for c in cluster_ids}

    tf_norm = tf_mat / (tf_mat.sum(axis=1, keepdims=True) + 1e-9)
    idf = np.log(len(docs) / ((tf_mat > 0).sum(axis=0) + 1)) + 1
    ctfidf = tf_norm * idf
    names = vec.get_feature_names_out()

    keywords = {
        cid: [names[j] for j in ctfidf[i].argsort()[-TOP_KEYWORDS:][::-1]]
        for i, cid in enumerate(cluster_ids)
    }
    distinct = {}
    for i, cid in enumerate(cluster_ids):
        top3 = ctfidf[i].argsort()[-3:][::-1]
        distinct[cid] = float(np.mean(idf[top3])) if len(top3) > 0 else 0.0
    return keywords, distinct


def main():
    print("── Load ──")
    comments_all = load_comments()
    scenes = load_scenes()
    clusters = load_clusters()
    print(f"  comments: {len(comments_all)}, scenes: {len(scenes)}, clusters file: {len(clusters)}")

    # 合併留言 embedding（cluster.py 沒有保留 embedding）
    df = clusters.merge(
        comments_all[["comment_id", "embedding"]],
        on="comment_id", how="left"
    )
    print(f"  merged: {len(df)} rows")

    X = np.stack(df["embedding"].values)
    scene_emb = np.stack(scenes["embedding"].values)

    print("\n── Scene similarity (留言 × 27 場景) ──")
    sim_matrix = compute_scene_sims(X, scene_emb)
    print(f"  sim_matrix shape: {sim_matrix.shape}, range [{sim_matrix.min():.3f}, {sim_matrix.max():.3f}]")

    # ── 每則留言對應到最近場景 ────────────────────────────────────────
    top1_idx = sim_matrix.argmax(axis=1)
    top1_sim = sim_matrix.max(axis=1)
    df["best_scene"] = [
        f"{scenes.iloc[i]['episode']}-{scenes.iloc[i]['scene_number']}"
        for i in top1_idx
    ]
    df["best_scene_sim"] = top1_sim

    # 儲存 comment-level 場景對應
    cmt_scene_path = DATA_DIR / "comment_scene_mapping.parquet"
    df[["comment_id", "text_clean", "like_count", "cluster_label",
        "best_scene", "best_scene_sim"]].to_parquet(cmt_scene_path, index=False)
    print(f"  Saved → {cmt_scene_path.name}")

    # ── Cluster-level analysis ──────────────────────────────────────────
    print("\n── Cluster analysis ──")
    valid = df[df["cluster_label"] != -1]
    cluster_ids = sorted(valid["cluster_label"].unique())
    print(f"  {len(cluster_ids)} valid clusters")

    keywords, distinctiveness = c_tf_idf(valid, cluster_ids)

    rows = []
    for cid in cluster_ids:
        grp_mask = (df["cluster_label"] == cid).values
        grp = df[grp_mask]
        size = len(grp)
        likes_total = int(grp["like_count"].sum())
        likes_max = int(grp["like_count"].max())

        # 1. engagement_density
        engage = float(np.log1p(likes_total / max(size, 1)))

        # 2. keyword_distinctiveness（已算）
        kw_dist = distinctiveness[cid]

        # 3. scene_specificity: cluster 內留言對最近場景的 sim 平均
        scn_spec = float(sim_matrix[grp_mask].max(axis=1).mean())

        # 4. variation_richness: 群內 embedding 對 centroid 的距離標準差
        X_g = X[grp_mask]
        X_n = X_g / (np.linalg.norm(X_g, axis=1, keepdims=True) + 1e-9)
        centroid = X_n.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        dist_to_c = 1.0 - X_n @ centroid
        var_rich = float(dist_to_c.std())

        # 主場景：cluster centroid 對 27 個場景的 sim
        centroid_4096 = np.stack(grp["embedding"].values).mean(axis=0)
        centroid_4096 = centroid_4096 / (np.linalg.norm(centroid_4096) + 1e-9)
        scene_sims_centroid = scene_emb / (np.linalg.norm(scene_emb, axis=1, keepdims=True) + 1e-9) @ centroid_4096
        best_scene_idx = int(np.argmax(scene_sims_centroid))
        sr = scenes.iloc[best_scene_idx]
        main_scene = f"{sr['episode']}-{sr['scene_number']}"
        main_scene_sim = float(scene_sims_centroid[best_scene_idx])

        # Top-3 高讚留言
        top3 = grp.nlargest(3, "like_count")["text_clean"].tolist()

        rows.append({
            "cluster_id": int(cid),
            "size": size,
            "likes_total": likes_total,
            "likes_max": likes_max,
            "engagement_density": engage,
            "kw_distinct": kw_dist,
            "scene_specificity": scn_spec,
            "variation_richness": var_rich,
            "main_scene": main_scene,
            "main_scene_sim": main_scene_sim,
            "keywords": ", ".join(keywords[cid][:TOP_KEYWORDS]),
            "top3_comments": " ||| ".join([str(t)[:100] for t in top3]),
        })

    out = pd.DataFrame(rows)

    # Normalize 各訊號到 0-1 (此次分析範圍內)
    def norm(s):
        a = np.array(s)
        return (a - a.min()) / (a.max() - a.min() + 1e-9)

    out["engage_n"] = norm(out["engagement_density"])
    out["kw_n"]     = norm(out["kw_distinct"])
    out["scene_n"]  = norm(out["scene_specificity"])
    out["var_n"]    = norm(out["variation_richness"])

    out["meme_quality"] = (
        W_ENGAGE * out["engage_n"]
      + W_KW     * out["kw_n"]
      + W_SCENE  * out["scene_n"]
      + W_VAR    * out["var_n"]
    )

    out = out.sort_values("meme_quality", ascending=False).reset_index(drop=True)

    out_path = DATA_DIR / "cluster_analysis.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  Saved → {out_path.name}  ({len(out)} clusters)")

    # Quick summary
    print("\n── Top 10 clusters by meme_quality ──")
    for _, r in out.head(10).iterrows():
        print(f"  C{r['cluster_id']:>3}  size={r['size']:>3}  likes={r['likes_total']:>6}  "
              f"quality={r['meme_quality']:.3f}  → {r['main_scene']}")
        print(f"         kw: {r['keywords'][:60]}")


if __name__ == "__main__":
    main()
