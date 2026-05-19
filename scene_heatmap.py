"""
scene_heatmap.py — 場景引爆熱度排行

聚合每個場景被多少 cluster 對應、總讚數、平均 sim，產出排行 + heatmap 圖。

需要先跑：
    cluster.py + analyze.py

Output:
    output/scene_heatmap.csv          排行表
    output/scene_heatmap.png          視覺化（top-20 cluster × 27 scenes）

Usage:
    uv run final_project/scene_heatmap.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_comments, load_scenes, load_clusters, OUTPUT_DIR, DATA_DIR


def main():
    print("── Load ──")
    analysis = pd.read_parquet(DATA_DIR / "cluster_analysis.parquet")
    clusters_df = load_clusters()
    scenes = load_scenes()
    print(f"  {len(analysis)} clusters, {len(scenes)} scenes")

    # 每個場景：被幾個 cluster 標為 main_scene？總讚數？
    print("\n── Aggregate by main_scene ──")
    by_scene = (
        analysis
        .groupby("main_scene")
        .agg(
            n_clusters     = ("cluster_id",   "count"),
            total_size     = ("size",         "sum"),
            total_likes    = ("likes_total",  "sum"),
            avg_meme_qual  = ("meme_quality", "mean"),
            avg_scene_sim  = ("main_scene_sim", "mean"),
        )
        .reset_index()
    )

    # 加 episode 排序
    by_scene["episode"] = by_scene["main_scene"].str.split("-").str[0]
    by_scene["scene_num"] = by_scene["main_scene"].str.split("-").str[1].astype(int)
    by_scene = by_scene.sort_values("total_likes", ascending=False).reset_index(drop=True)

    # ── Top clusters 名稱（每場景前 3 個 cluster） ──
    top_per_scene = {}
    for scene_label in by_scene["main_scene"]:
        clusters_in_scene = analysis[analysis["main_scene"] == scene_label].nlargest(3, "likes_total")
        names = [f"C{int(r['cluster_id'])}({r['likes_total']})" for _, r in clusters_in_scene.iterrows()]
        top_per_scene[scene_label] = " / ".join(names)
    by_scene["top_clusters"] = by_scene["main_scene"].map(top_per_scene)

    out_path = OUTPUT_DIR / "scene_heatmap.csv"
    by_scene.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved → {out_path}")

    print("\n── Top 10 hottest scenes ──")
    for _, r in by_scene.head(10).iterrows():
        print(f"  {r['main_scene']:>10}  clusters={r['n_clusters']:>2}  "
              f"likes={r['total_likes']:>6}  quality={r['avg_meme_qual']:.3f}")
        print(f"               top: {r['top_clusters']}")

    # ── Heatmap 視覺化 ──
    print("\n── Heatmap ──")
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # 試載入中文字型
        cn_fonts = [f.name for f in fm.fontManager.ttflist if any(
            kw in f.name for kw in ["Microsoft JhengHei", "SimHei", "NotoSansCJK", "WenQuanYi"]
        )]
        if cn_fonts:
            matplotlib.rcParams["font.family"] = cn_fonts[0]
        matplotlib.rcParams["axes.unicode_minus"] = False

        # 載入 comments 和 cluster centroid → 每個 scene 的對應強度
        comments_all = load_comments()
        df = clusters_df.merge(comments_all[["comment_id", "embedding"]], on="comment_id", how="left")
        df = df[df["cluster_label"] != -1]

        # Top 20 cluster
        top20 = analysis.head(20)["cluster_id"].tolist()

        # 算每個 top cluster 對 27 scene 的 sim
        scene_emb = np.stack(scenes["embedding"].values)
        scene_emb = scene_emb / (np.linalg.norm(scene_emb, axis=1, keepdims=True) + 1e-9)

        heatmap_data = []
        for cid in top20:
            grp = df[df["cluster_label"] == cid]
            embs = np.stack(grp["embedding"].values)
            c = embs.mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-9)
            sims = scene_emb @ c
            heatmap_data.append(sims)

        heatmap = np.array(heatmap_data)
        scene_labels = [f"{r['episode']}-{r['scene_number']}" for _, r in scenes.iterrows()]
        cluster_labels = [f"C{c}" for c in top20]

        fig, ax = plt.subplots(figsize=(16, 8))
        im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(cluster_labels)))
        ax.set_yticklabels(cluster_labels, fontsize=9)
        ax.set_xticks(range(len(scene_labels)))
        ax.set_xticklabels(scene_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title("Top-20 Clusters × 27 Scenes — Cosine Similarity Heatmap", fontsize=12)
        plt.colorbar(im, ax=ax, label="cosine sim")
        plt.tight_layout()

        png_path = OUTPUT_DIR / "scene_heatmap.png"
        plt.savefig(png_path, dpi=120)
        plt.close()
        print(f"  Saved → {png_path}")
    except ImportError:
        print("  matplotlib not available — skip heatmap PNG")


if __name__ == "__main__":
    main()
