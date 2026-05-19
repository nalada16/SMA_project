"""
SMA Meme Discovery — clustering pipeline v2 (autoresearch experiment version).
Agent 修改 TUNEABLE PARAMETERS 區塊。

Usage (from repo root):
    uv run python -X utf8 SMA/experiment/cluster.py > SMA/experiment/run.log 2>&1
    grep "^L1_pass:\\|^recall:\\|^scene_acc:" SMA/experiment/run.log
"""
import sys
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# SMA/ 根目錄（prepare_sma.py 在那裡）
SMA_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SMA_DIR))
sys.path.insert(0, str(Path(__file__).parent))  # stages/ 在 experiment/ 下

from prepare_sma import load_comments, load_scenes, load_ground_truth, evaluate_layer1
from stages.prefilter    import apply_prefilter
from stages.embed_aug    import compute_scene_sims, apply_embed_aug
from stages.dim_reduce   import apply_dim_reduce
from stages.cluster_algo import apply_cluster
from stages.postprocess  import apply_postprocess
from stages.meme_score   import score_clusters

CACHE_DIR = SMA_DIR / ".cache"
EXP_DIR   = SMA_DIR / "data" / "experiments"
CACHE_DIR.mkdir(exist_ok=True)
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# TUNEABLE PARAMETERS — autoresearch agent 修改這裡
# ══════════════════════════════════════════════════════════════════════
PREFILTER       = "F2"
EMBED_AUG       = "E0"
DIM_REDUCE      = "DR0"
CLUSTER_ALGO    = "C0"
POSTPROCESS     = "P3"
MEME_SCORE      = "S3"

UMAP_N_COMPONENTS = 50
UMAP_N_NEIGHBORS  = 15
UMAP_MIN_DIST     = 0.0
MIN_CLUSTER_SIZE  = 20
MIN_SAMPLES       = 1
HDBSCAN_METHOD    = "eom"
AGGLO_DIST_THRESHOLD = 0.5
KMEANS_K_RANGE = (15, 40)
F_MIN_SCENE_SIM = 0.3
F_MIN_LIKES     = 2
F_LEN_RANGE     = (5, 300)
MERGE_DIST_THRESHOLD     = 0.3
NOISE_RESCUE_SIM         = 0.5
INTERACTIVE_MIN_LIKES    = 500
TOP_K_CLUSTERS_FOR_REVIEW = 15
# ══════════════════════════════════════════════════════════════════════


def cache_key():
    params = dict(
        pre=PREFILTER, ea=EMBED_AUG, dr=DIM_REDUCE,
        nc=UMAP_N_COMPONENTS, nn=UMAP_N_NEIGHBORS, md=UMAP_MIN_DIST,
        msc=F_MIN_SCENE_SIM, ml=F_MIN_LIKES,
    )
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def current_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=SMA_DIR.parent   # repo root
        ).decode().strip()
    except Exception:
        return "nogit"


def write_review(df, scored, layer1, interactive_candidates, scenes, scene_sims,
                 commit, pipeline_str, out_path):
    lines = []
    lines.append(f"# Experiment {commit}")
    lines.append(f"Pipeline: {pipeline_str}")
    lines.append(f"Layer 1: recall={layer1['recall_on_known']:.3f}, "
                 f"scene_accuracy={layer1['scene_accuracy']:.3f}, "
                 f"passed={'YES' if layer1['passed'] else 'NO'}")
    lines.append(f"Clusters: {len(scored)}, "
                 f"Noise: {(df['cluster_label'] == -1).sum()}, "
                 f"Total valid: {len(df)}")
    lines.append("")
    lines.append("## Ground Truth Check\n")
    for d in layer1["details"]:
        status = "✓" if d["found"] and d["scene_correct"] else (
                 "△" if d["found"] else "✗")
        line = f"- {status} **{d['id']}** ({d['name']}): "
        if d["found"]:
            line += f"cluster={d['cluster_id']}, scene→{d['best_scene']}"
            if not d["scene_correct"]:
                line += f" (claimed: {d.get('claimed_scene')})"
        else:
            line += f"NOT FOUND ({d.get('reason', '')})"
        lines.append(line)
    lines.append("")

    lines.append(f"## Top {TOP_K_CLUSTERS_FOR_REVIEW} Clusters by Meme Quality\n")
    valid_df = df[df["cluster_label"] != -1]
    gt_clusters = {d["cluster_id"]: d["id"] for d in layer1["details"]
                   if d["found"] and d["cluster_id"] is not None}

    for _, row in scored.head(TOP_K_CLUSTERS_FOR_REVIEW).iterrows():
        cid = int(row["cluster_id"])
        grp = valid_df[valid_df["cluster_label"] == cid]
        top5 = grp.nlargest(5, "like_count")
        cluster_mask = (df["cluster_label"] == cid).values
        scene_sim_avg = scene_sims[cluster_mask].mean(axis=0)
        best_scene_idx = int(np.argmax(scene_sim_avg))
        sr = scenes.iloc[best_scene_idx]
        scene_label = f"{sr['episode']}-{sr['scene_number']}"
        scene_sim_value = float(scene_sim_avg[best_scene_idx])
        gt_tag = f"  ← GT match: **{gt_clusters[cid]}**" if cid in gt_clusters else ""
        lines.append(f"### Cluster {cid}  size={row['size']}  likes={row['likes_total']}{gt_tag}")
        lines.append(f"- meme_quality={row['meme_quality']:.3f}  "
                     f"(eng={row['engagement_n']:.2f}, kw={row['kw_n']:.2f}, "
                     f"scn={row['scene_n']:.2f}, var={row['var_n']:.2f})")
        lines.append(f"- centroid → {scene_label} (sim={scene_sim_value:.3f})")
        lines.append(f"- keywords: {row['keywords']}")
        lines.append(f"- Top 5 留言:")
        for i, (_, r) in enumerate(top5.iterrows(), 1):
            text = str(r["text_clean"])[:120].replace("\n", " ")
            lines.append(f"  {i}. [{int(r['like_count']):>5}讚] {text}")
        lines.append("")

    if interactive_candidates:
        lines.append("## Interactive Meme Candidates (P3 detector)\n")
        for cand in interactive_candidates[:15]:
            text = str(cand["text"])[:140].replace("\n", " ")
            lines.append(f"- [{cand['likes']:>5}讚] {text}")
        lines.append("")
    else:
        noise = df[df["cluster_label"] == -1].nlargest(15, "like_count")
        if len(noise) > 0:
            lines.append("## Noise High-Likes (top 15)\n")
            for _, r in noise.iterrows():
                text = str(r["text_clean"])[:140].replace("\n", " ")
                lines.append(f"- [{int(r['like_count']):>5}讚] {text}")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run():
    pipeline_str = f"{PREFILTER}/{EMBED_AUG}/{DIM_REDUCE}/{CLUSTER_ALGO}/{POSTPROCESS}/{MEME_SCORE}"
    commit = current_commit()
    print("── Load ──")
    df_full = load_comments()
    scenes = load_scenes()
    gt = load_ground_truth()
    print(f"  {len(df_full)} valid comments, {len(scenes)} scenes")

    X_full = np.stack(df_full["embedding"].values)
    scene_emb = np.stack(scenes["embedding"].values)
    print("── Scene similarity ──")
    scene_sims_full = compute_scene_sims(X_full, scene_emb)

    print(f"── Pipeline: {pipeline_str} ──")
    df, keep_mask = apply_prefilter(
        df_full, scene_sims_full, PREFILTER,
        min_scene_sim=F_MIN_SCENE_SIM, min_likes=F_MIN_LIKES,
        len_range=F_LEN_RANGE,
    )
    X = X_full[keep_mask]
    scene_sims = scene_sims_full[keep_mask]
    df = df.reset_index(drop=True)
    print(f"  [{PREFILTER}] {len(df)} after filter")

    X_aug = apply_embed_aug(X, scene_sims, df["like_count"].values, EMBED_AUG)
    print(f"  [{EMBED_AUG}] dim: {X_aug.shape}")

    cache_path = CACHE_DIR / f"reduced_{cache_key()}.npy"
    if cache_path.exists() and DIM_REDUCE != "DR4":
        print(f"  [{DIM_REDUCE}] cache hit")
        X_red = np.load(cache_path)
    else:
        X_red = apply_dim_reduce(
            X_aug, DIM_REDUCE,
            umap_n_components=UMAP_N_COMPONENTS,
            umap_n_neighbors=UMAP_N_NEIGHBORS,
            umap_min_dist=UMAP_MIN_DIST,
        )
        if DIM_REDUCE != "DR4":
            np.save(cache_path, X_red)
        print(f"  [{DIM_REDUCE}] shape: {X_red.shape}")

    labels = apply_cluster(
        X_red, CLUSTER_ALGO,
        min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
        hdbscan_method=HDBSCAN_METHOD,
        aggl_dist_threshold=AGGLO_DIST_THRESHOLD,
        kmeans_k_range=KMEANS_K_RANGE,
    )
    df["cluster_label"] = labels
    print(f"  [{CLUSTER_ALGO}] n_clusters={len(set(labels))-(1 if -1 in labels else 0)}")

    df, interactive_candidates = apply_postprocess(
        df, X_red, POSTPROCESS,
        merge_dist_threshold=MERGE_DIST_THRESHOLD,
        noise_rescue_sim=NOISE_RESCUE_SIM,
        interactive_min_likes=INTERACTIVE_MIN_LIKES,
    )
    final_labels = df["cluster_label"].values
    n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
    noise_ratio = (final_labels == -1).sum() / len(final_labels)
    print(f"  [{POSTPROCESS}] n_clusters={n_clusters}, noise={noise_ratio:.2f}, "
          f"interactive={len(interactive_candidates)}")

    scored = score_clusters(df, X_red, scene_sims, scenes, MEME_SCORE)
    print(f"  [{MEME_SCORE}] scored {len(scored)} clusters")

    layer1 = evaluate_layer1(df, X_red, scenes, gt)

    # ── metric output — do NOT change this format (grep target) ───────
    print("\n---")
    print(f"L1_pass:        {1 if layer1['passed'] else 0}")
    print(f"L1_score:       {layer1['l1_score']:.3f}")
    print(f"recall:         {layer1['recall_on_known']:.3f}")
    print(f"scene_acc:      {layer1['scene_accuracy']:.3f}")
    print(f"silhouette:     {layer1['silhouette']:.3f}")
    print(f"n_clusters:     {n_clusters}")
    print(f"noise_ratio:    {noise_ratio:.2f}")
    print(f"pipeline:       {pipeline_str}")
    print(f"commit:         {commit}")
    print("---\n")
    # ──────────────────────────────────────────────────────────────────

    review_path = EXP_DIR / f"{commit}_clusters_for_review.md"
    write_review(df, scored, layer1, interactive_candidates, scenes, scene_sims,
                 commit, pipeline_str, review_path)
    print(f"Review → {review_path}")


if __name__ == "__main__":
    run()
