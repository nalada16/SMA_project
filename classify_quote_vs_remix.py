"""
classify_quote_vs_remix.py — 把每個 cluster 分類成「引用 vs 改編 vs 二創」

做法：
    每個 cluster 的高讚留言 → 對所有劇本台詞做 cosine similarity → 取最高分
    sim > 0.75:        direct_quote
    0.5 < sim ≤ 0.75:  template_modification
    sim ≤ 0.5:         creative_derivative

需要先跑：
    cluster.py        → clusters.parquet
    embed_lines.py    → lines_with_embedding.parquet
    analyze.py        → cluster_analysis.parquet

Output:
    output/quote_classification.csv

Usage:
    uv run final_project/classify_quote_vs_remix.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_comments, load_clusters, load_lines, OUTPUT_DIR, DATA_DIR

THRESHOLD_DIRECT     = 0.75
THRESHOLD_TEMPLATE   = 0.50
TOP_COMMENTS_PER_CLUSTER = 5   # 用 cluster 內 top N 高讚留言判斷


def classify(sim: float) -> str:
    if sim > THRESHOLD_DIRECT:
        return "direct_quote"
    if sim > THRESHOLD_TEMPLATE:
        return "template_modification"
    return "creative_derivative"


def main():
    print("── Load ──")
    comments_all = load_comments()
    clusters_df = load_clusters()
    lines = load_lines()
    print(f"  comments: {len(comments_all)}, lines: {len(lines)}")

    # 合併 cluster label + embedding
    df = clusters_df.merge(
        comments_all[["comment_id", "embedding"]],
        on="comment_id", how="left"
    )

    line_emb = np.stack(lines["embedding"].values)
    line_emb = line_emb / (np.linalg.norm(line_emb, axis=1, keepdims=True) + 1e-9)

    print("\n── Classify clusters ──")
    valid = df[df["cluster_label"] != -1]
    cluster_ids = sorted(valid["cluster_label"].unique())

    rows = []
    for cid in cluster_ids:
        grp = valid[valid["cluster_label"] == cid].nlargest(TOP_COMMENTS_PER_CLUSTER, "like_count")
        size_full = (valid["cluster_label"] == cid).sum()
        likes_total = int(valid[valid["cluster_label"] == cid]["like_count"].sum())

        # 取 top N 留言對所有 line 算 sim，取最大
        cmt_emb = np.stack(grp["embedding"].values)
        cmt_emb = cmt_emb / (np.linalg.norm(cmt_emb, axis=1, keepdims=True) + 1e-9)
        sims = cmt_emb @ line_emb.T    # (N, n_lines)

        # 每則留言對最接近的台詞
        best_sims = sims.max(axis=1)
        best_idxs = sims.argmax(axis=1)
        avg_top_sim = float(best_sims.mean())
        max_top_sim = float(best_sims.max())

        # 用平均判斷整群類型，但也記錄個別留言的對應
        cluster_type = classify(avg_top_sim)

        # 最高分留言對應的台詞
        argmax_in_grp = int(np.argmax(best_sims))
        best_line_idx = best_idxs[argmax_in_grp]
        best_line = lines.iloc[best_line_idx]
        best_comment = grp.iloc[argmax_in_grp]

        rows.append({
            "cluster_id":        int(cid),
            "size":              int(size_full),
            "likes_total":       likes_total,
            "type":              cluster_type,
            "avg_top_sim":       avg_top_sim,
            "max_top_sim":       max_top_sim,
            "best_comment":      str(best_comment["text_clean"])[:120],
            "best_comment_likes": int(best_comment["like_count"]),
            "matched_line":      str(best_line["text"])[:120],
            "matched_character": str(best_line["character"]),
            "matched_scene":     f"{best_line['episode']}-{best_line['scene_number']}",
        })

    out = pd.DataFrame(rows).sort_values("likes_total", ascending=False)

    out_path = OUTPUT_DIR / "quote_classification.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved → {out_path}")

    # ── Summary ──
    print("\n── Type distribution ──")
    print(out["type"].value_counts().to_string())

    print("\n── Top 5 direct_quote clusters ──")
    direct = out[out["type"] == "direct_quote"].head(5)
    for _, r in direct.iterrows():
        print(f"  C{r['cluster_id']:>3}  likes={r['likes_total']:>5}  "
              f"sim={r['avg_top_sim']:.3f}")
        print(f"        留言：{r['best_comment'][:70]}")
        print(f"        台詞：{r['matched_character']}：{r['matched_line'][:70]}")

    print("\n── Top 5 template_modification ──")
    tmpl = out[out["type"] == "template_modification"].head(5)
    for _, r in tmpl.iterrows():
        print(f"  C{r['cluster_id']:>3}  likes={r['likes_total']:>5}  "
              f"sim={r['avg_top_sim']:.3f}")
        print(f"        留言：{r['best_comment'][:70]}")
        print(f"        台詞：{r['matched_character']}：{r['matched_line'][:70]}")

    print("\n── Top 5 creative_derivative ──")
    creative = out[out["type"] == "creative_derivative"].head(5)
    for _, r in creative.iterrows():
        print(f"  C{r['cluster_id']:>3}  likes={r['likes_total']:>5}  "
              f"sim={r['avg_top_sim']:.3f}")
        print(f"        留言：{r['best_comment'][:70]}")


if __name__ == "__main__":
    main()
