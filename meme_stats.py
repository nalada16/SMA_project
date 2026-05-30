"""
meme_stats.py — 從最新 meme_inventory_N.yaml 產出迷因統計分析與圖表。

分析項目（各項均輸出「迷因」、「非迷因」、「合計」三份）：
    1. 高頻場景   哪一幕出現最多 cluster，加上讚數
    2. 高頻角色   哪個角色出現在最多 cluster 對應場景，加上讚數

僅限「迷因」的額外分析：
    3. 迷因類型分布      + 讚數
    4. 商業價值分布      + 讚數
    5. 引用 vs 改編分布  + 讚數（需要 quote_classification.csv）

圖表輸出：
    output/meme_chart_characters.png   角色分組橫條（迷因 vs 非迷因）
    output/meme_chart_scenes.png       場景堆疊橫條（迷因 + 非迷因）
    output/meme_chart_business.png     商業價值圓餅（迷因專用）
    output/meme_chart_types.png        迷因類型橫條（迷因專用）

CSV 輸出（各有 _meme / _non_meme / _all 三份）：
    output/meme_stats_scenes_*.csv
    output/meme_stats_characters_*.csv
    output/meme_stats_types.csv
    output/meme_stats_business.csv
    output/meme_stats_quote_types.csv

Usage:
    uv run final_project/meme_stats.py
    uv run final_project/meme_stats.py --inventory output/meme_inventory_1.yaml
    uv run final_project/meme_stats.py --top 20
    uv run final_project/meme_stats.py --no-charts
"""
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, str(Path(__file__).parent))
from prepare import DATA_DIR, OUTPUT_DIR, load_lines


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def find_latest_inventory(output_dir: Path) -> Path:
    numbered = [
        p for p in output_dir.glob("meme_inventory_*.yaml")
        if p.stem.split("_")[-1].isdigit()
    ]
    if not numbered:
        raise FileNotFoundError(f"找不到 meme_inventory_N.yaml 在 {output_dir}")
    return max(numbered, key=lambda p: int(p.stem.split("_")[-1]))


def load_inventory(path: Path) -> tuple[list[dict], list[dict]]:
    """回傳 (memes, non_memes)。"""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    memes = []
    for key in ("discovered_memes", "confirmed_memes", "algorithm_discovered"):
        memes.extend(data.get(key) or [])

    non_memes = list(data.get("non_meme_clusters") or [])
    return memes, non_memes


def build_df(items: list[dict], group: str,
             analysis: pd.DataFrame) -> pd.DataFrame:
    """把 meme 或 non_meme list 轉成 DataFrame，join cluster 讚數。"""
    rows = []
    for item in items:
        cid = item.get("cluster_id")
        if cid is None:
            continue
        cid = int(cid)
        row = {
            "cluster_id":     cid,
            "group":          group,
            "name":           item.get("name") or item.get("theme", ""),
            "meme_type":      item.get("type", ""),
            "business_value": item.get("business_value", ""),
        }
        ca = analysis[analysis["cluster_id"] == cid]
        if not ca.empty:
            r = ca.iloc[0]
            row["main_scene"]   = r["main_scene"]
            row["likes_total"]  = int(r["likes_total"])
            row["likes_max"]    = int(r["likes_max"])
            row["cluster_size"] = int(r["size"])
        else:
            row.update(main_scene=None, likes_total=0, likes_max=0, cluster_size=0)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["cluster_id", "group", "name", "meme_type", "business_value",
                 "main_scene", "likes_total", "likes_max", "cluster_size"]
    )


def compute_scene_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df["main_scene"].isna().all():
        return pd.DataFrame()
    return (
        df[df["main_scene"].notna()]
        .groupby("main_scene")
        .agg(
            cluster_count = ("name", "count"),
            總讚數        = ("likes_total", "sum"),
            最高讚        = ("likes_max", "max"),
            名稱列表      = ("name", lambda x: "  /  ".join(x)),
        )
        .sort_values(["cluster_count", "總讚數"], ascending=False)
        .reset_index()
        .rename(columns={"main_scene": "場景", "cluster_count": "cluster 數"})
    )


def compute_char_stats(df: pd.DataFrame,
                       scene_chars: dict[str, list[str]]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    char_count: dict[str, int] = defaultdict(int)
    char_likes: dict[str, int] = defaultdict(int)
    for _, row in df.iterrows():
        scene = row["main_scene"]
        if not scene or pd.isna(scene):
            continue
        for char in scene_chars.get(scene, []):
            char_count[char] += 1
            char_likes[char] += row["likes_total"]
    if not char_count:
        return pd.DataFrame()
    return (
        pd.DataFrame({
            "角色":        list(char_count.keys()),
            "出現 cluster 數": list(char_count.values()),
            "累積讚數":    [char_likes[c] for c in char_count],
        })
        .sort_values(["出現 cluster 數", "累積讚數"], ascending=False)
        .reset_index(drop=True)
    )


def sep(title: str):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print('─'*62)


def show(df: pd.DataFrame, top_n: int = None):
    if df.empty:
        print("  （無資料）")
        return
    print((df.head(top_n) if top_n else df).to_string(index=False))


def save_csv(df: pd.DataFrame, name: str):
    path = OUTPUT_DIR / f"meme_stats_{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  ✓ {path.name}")


# ══════════════════════════════════════════════════════════════════════
# Charts
# ══════════════════════════════════════════════════════════════════════

def get_cjk_font() -> str | None:
    candidates = [
        "Microsoft JhengHei", "Microsoft YaHei",
        "PingFang TC", "PingFang SC",
        "Noto Sans CJK TC", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei", "SimHei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def make_charts(meme_scene, non_meme_scene,
                meme_char, non_meme_char,
                type_agg, biz_agg, top_n: int):
    font = get_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False

    BG      = "#FAFAFA"
    C_MEME  = "#4E79A7"   # 藍：迷因
    C_NON   = "#BAB0AC"   # 灰：非迷因
    C_ACC   = "#F28E2B"   # 橘：accent
    PIE_CLR = ["#59A14F", "#F28E2B", "#E15759"]

    # ── 1. 角色分組橫條（迷因 vs 非迷因）──────────────────────────────
    # 合併兩份，取 union of characters
    all_chars = set(meme_char.get("角色", [])) | set(non_meme_char.get("角色", []))
    if all_chars:
        mc = meme_char.set_index("角色")["累積讚數"].to_dict() if not meme_char.empty else {}
        nc = non_meme_char.set_index("角色")["累積讚數"].to_dict() if not non_meme_char.empty else {}
        combined = pd.DataFrame({
            "角色":   sorted(all_chars),
            "迷因":   [mc.get(c, 0) for c in sorted(all_chars)],
            "非迷因": [nc.get(c, 0) for c in sorted(all_chars)],
        })
        combined["total"] = combined["迷因"] + combined["非迷因"]
        combined = combined.nlargest(top_n, "total").sort_values("total")

        fig, ax = plt.subplots(figsize=(9, max(4, len(combined) * 0.5)))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        y = range(len(combined))
        h = 0.35
        ax.barh([yi + h/2 for yi in y], combined["迷因"],   height=h,
                color=C_MEME, label="迷因")
        ax.barh([yi - h/2 for yi in y], combined["非迷因"], height=h,
                color=C_NON,  label="非迷因")
        ax.set_yticks(list(y))
        ax.set_yticklabels(combined["角色"].tolist())
        ax.set_xlabel("累積讚數")
        ax.set_title("角色 × 累積讚數（迷因 vs 非迷因場景）", fontsize=13, pad=10)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out = OUTPUT_DIR / "meme_chart_characters.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {out.name}")

    # ── 2. 場景堆疊橫條 ─────────────────────────────────────────────
    all_scenes = set(meme_scene.get("場景", [])) | set(non_meme_scene.get("場景", []))
    if all_scenes:
        ms = meme_scene.set_index("場景")["總讚數"].to_dict()     if not meme_scene.empty else {}
        ns = non_meme_scene.set_index("場景")["總讚數"].to_dict() if not non_meme_scene.empty else {}
        sc = pd.DataFrame({
            "場景":   sorted(all_scenes),
            "迷因":   [ms.get(s, 0) for s in sorted(all_scenes)],
            "非迷因": [ns.get(s, 0) for s in sorted(all_scenes)],
        })
        sc["total"] = sc["迷因"] + sc["非迷因"]
        sc = sc.nlargest(top_n, "total").sort_values("total")

        fig, ax = plt.subplots(figsize=(9, max(4, len(sc) * 0.48)))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.barh(sc["場景"], sc["迷因"],   color=C_MEME, label="迷因",   height=0.6)
        ax.barh(sc["場景"], sc["非迷因"], left=sc["迷因"],
                color=C_NON,  label="非迷因", height=0.6)
        ax.set_xlabel("cluster 總讚數")
        ax.set_title("場景 × 讚數（迷因 + 非迷因堆疊）", fontsize=13, pad=10)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out = OUTPUT_DIR / "meme_chart_scenes.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {out.name}")

    # ── 3. 商業價值圓餅（迷因專用）──────────────────────────────────
    if not biz_agg.empty:
        order   = ["HIGH", "MEDIUM", "LOW"]
        biz_plt = biz_agg[biz_agg["商業價值"].isin(order)].set_index("商業價值").reindex(order).dropna()
        sizes   = biz_plt["迷因數"].values
        labels  = [f"{v}\n{n} 個  /  {l:,} 讚"
                   for v, n, l in zip(biz_plt.index, biz_plt["迷因數"], biz_plt["總讚數"])]
        colors  = PIE_CLR[: len(sizes)]
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(BG)
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.0f%%", startangle=90,
            pctdistance=0.75, labeldistance=1.18,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )
        for at in autotexts:
            at.set_fontsize(10)
        ax.set_title("商業價值分布（迷因）", fontsize=13, pad=12)
        plt.tight_layout()
        out = OUTPUT_DIR / "meme_chart_business.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {out.name}")

    # ── 4. 迷因類型橫條（迷因專用）──────────────────────────────────
    if not type_agg.empty:
        df = type_agg.sort_values("總讚數")
        fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.5)))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        bars = ax.barh(df["迷因類型"], df["總讚數"], color=C_ACC, height=0.6)
        for bar, cnt in zip(bars, df["迷因數"]):
            ax.text(bar.get_width() + max(df["總讚數"]) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(bar.get_width()):,}  ({cnt} 個)",
                    va="center", fontsize=9)
        ax.set_xlabel("迷因 cluster 總讚數")
        ax.set_title("迷因類型 × 總讚數", fontsize=13, pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out = OUTPUT_DIR / "meme_chart_types.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=str, default=None)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()

    # ── Load ────────────────────────────────────────────────────────
    inv_path = Path(args.inventory) if args.inventory else find_latest_inventory(OUTPUT_DIR)
    print(f"\n📂  Inventory : {inv_path.name}")

    memes, non_memes = load_inventory(inv_path)
    analysis = pd.read_parquet(DATA_DIR / "cluster_analysis.parquet")
    lines    = load_lines()

    quote_path = OUTPUT_DIR / "quote_classification.csv"
    quote_df   = pd.read_csv(quote_path) if quote_path.exists() else None

    print(f"    discovered memes   : {len(memes)}")
    print(f"    non-meme clusters  : {len(non_memes)}")

    # ── Scene → characters ──────────────────────────────────────────
    lines["scene_key"] = (
        lines["episode"].astype(str) + "-" + lines["scene_number"].astype(str)
    )
    scene_chars: dict[str, list[str]] = (
        lines.groupby("scene_key")["character"]
        .apply(lambda x: sorted(x.unique()))
        .to_dict()
    )

    # ── Build DataFrames ────────────────────────────────────────────
    meme_df     = build_df(memes,     "meme",     analysis)
    non_meme_df = build_df(non_memes, "non_meme", analysis)
    all_df      = pd.concat([meme_df, non_meme_df], ignore_index=True)

    if quote_df is not None and not meme_df.empty:
        meme_df = meme_df.merge(
            quote_df[["cluster_id", "type"]].rename(columns={"type": "quote_type"}),
            on="cluster_id", how="left",
        )
    else:
        if not meme_df.empty:
            meme_df["quote_type"] = None

    # ── Scene stats ─────────────────────────────────────────────────
    meme_scene     = compute_scene_stats(meme_df)
    non_meme_scene = compute_scene_stats(non_meme_df)
    all_scene      = compute_scene_stats(all_df)

    sep("【高頻場景 — 迷因】")
    show(meme_scene, args.top)
    sep("【高頻場景 — 非迷因】")
    show(non_meme_scene, args.top)
    sep("【高頻場景 — 合計】")
    show(all_scene, args.top)

    # ── Character stats ─────────────────────────────────────────────
    meme_char     = compute_char_stats(meme_df,     scene_chars)
    non_meme_char = compute_char_stats(non_meme_df, scene_chars)
    all_char      = compute_char_stats(all_df,      scene_chars)

    sep("【高頻角色 — 迷因】")
    show(meme_char, args.top)
    sep("【高頻角色 — 非迷因】")
    show(non_meme_char, args.top)
    sep("【高頻角色 — 合計】")
    show(all_char, args.top)

    # ── Meme-only stats ─────────────────────────────────────────────
    type_agg = pd.DataFrame()
    biz_agg  = pd.DataFrame()
    qt_agg   = None

    if not meme_df.empty:
        type_agg = (
            meme_df.groupby("meme_type")
            .agg(迷因數=("name", "count"),
                 總讚數=("likes_total", "sum"),
                 平均讚=("likes_total", "mean"),
                 最高讚=("likes_max", "max"))
            .sort_values("總讚數", ascending=False)
            .reset_index()
            .rename(columns={"meme_type": "迷因類型"})
        )
        type_agg["平均讚"] = type_agg["平均讚"].round(0).astype(int)
        sep("【迷因類型分布】")
        show(type_agg)

        biz_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        biz_agg = (
            meme_df.groupby("business_value")
            .agg(迷因數=("name", "count"),
                 總讚數=("likes_total", "sum"),
                 平均讚=("likes_total", "mean"),
                 迷因列表=("name", lambda x: "  /  ".join(x)))
            .reset_index()
            .rename(columns={"business_value": "商業價值"})
        )
        biz_agg["平均讚"] = biz_agg["平均讚"].round(0).astype(int)
        biz_agg["_o"] = biz_agg["商業價值"].map(biz_order).fillna(9)
        biz_agg = biz_agg.sort_values("_o").drop(columns="_o").reset_index(drop=True)
        sep("【商業價值分布】")
        show(biz_agg[["商業價值", "迷因數", "總讚數", "平均讚", "迷因列表"]])

        if "quote_type" in meme_df.columns and meme_df["quote_type"].notna().any():
            qt_agg = (
                meme_df.groupby("quote_type")
                .agg(迷因數=("name", "count"),
                     總讚數=("likes_total", "sum"),
                     平均讚=("likes_total", "mean"),
                     最高讚=("likes_max", "max"))
                .sort_values("迷因數", ascending=False)
                .reset_index()
                .rename(columns={"quote_type": "留言類型"})
            )
            qt_agg["平均讚"] = qt_agg["平均讚"].round(0).astype(int)
            sep("【引用 vs 改編分布】")
            show(qt_agg)

    # ── Save CSV ────────────────────────────────────────────────────
    sep("儲存 CSV")
    save_csv(meme_scene,     "scenes_meme")
    save_csv(non_meme_scene, "scenes_non_meme")
    save_csv(all_scene,      "scenes_all")
    save_csv(meme_char,      "characters_meme")
    save_csv(non_meme_char,  "characters_non_meme")
    save_csv(all_char,       "characters_all")
    if not type_agg.empty:
        save_csv(type_agg, "types")
    if not biz_agg.empty:
        save_csv(biz_agg,  "business")
    if qt_agg is not None:
        save_csv(qt_agg,   "quote_types")

    # ── Charts ──────────────────────────────────────────────────────
    if not args.no_charts:
        sep("產生圖表")
        make_charts(meme_scene, non_meme_scene,
                    meme_char,  non_meme_char,
                    type_agg, biz_agg, top_n=args.top)

    print(f"\n{'─'*62}")
    print("完成。")


if __name__ == "__main__":
    main()
