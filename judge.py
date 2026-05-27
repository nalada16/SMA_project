"""
judge.py — 產出半自動的 LLM judge review 文件。

這個腳本本身不呼叫 LLM。它做的事：
    1. 讀 cluster_analysis.parquet
    2. 整理 top-K cluster 的留言、關鍵字、場景
    3. 把 ground_truth 當作 few-shot 範例放在前面
    4. 寫一份 review_for_judge.md
    5. 同時嵌入「給 agent 的判斷 prompt」，未來任何 agent 開檔即可上手

之後流程：
    - 開啟 Claude Code (或人類)
    - 讀 final_project/output/review_for_judge.md
    - 照檔案末尾的 prompt 寫 final_project/output/meme_inventory_N.yaml（序號自動遞增）

Usage:
    uv run final_project/judge.py
    uv run final_project/judge.py --top-k 20   # 改判斷的 cluster 數量
"""
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_comments, load_clusters, load_ground_truth, OUTPUT_DIR, DATA_DIR

DEFAULT_TOP_K = 20            # 排前 K 名的 cluster 拿去判斷
NOISE_TOP_N   = 20            # noise 中高讚留言前 N 個拿去判斷


def render_few_shot(gt: dict) -> str:
    """把 ground_truth.yaml 內容渲染成 few-shot 範例（風格示範，非答案）。"""
    memes = gt["confirmed_memes"]
    lines = []
    lines.append("## Few-shot 範例（迷因風格示範）\n")
    lines.append(f"以下 {len(memes)} 個是人類認定的迷因或具商業價值的範例。"
                 f"請依此風格在待判斷的 cluster / noise 留言中找出同類內容，"
                 f"不需要逐一回對這些範例：\n")

    for meme in memes:
        lines.append(f"### {meme['id']} — {meme['name']}")
        likes = meme.get("canonical_likes")
        if likes is not None:
            lines.append(f"- **代表留言**：「{meme['canonical_text']}」（{likes} 讚）")
        else:
            lines.append(f"- **代表留言**：「{meme['canonical_text']}」")
        lines.append(f"- **類型**：`{meme['type']}`")
        if meme.get("usage_note"):
            lines.append(f"- **使用情境**：{meme['usage_note']}")
        lines.append("")

    lines.append("### potential 範例")
    for p in gt.get("potential_memes", []):
        lines.append(f"- **{p['id']}** 「{p['canonical_text']}」（{p['canonical_likes']} 讚）"
                     f" → `{p['type']}`，{p.get('note', '')}")
    lines.append("")

    lines.append("---\n")
    return "\n".join(lines)


def render_clusters(analysis: pd.DataFrame, comments: pd.DataFrame, top_k: int) -> str:
    """渲染 top-K cluster 的判斷材料。"""
    lines = []
    lines.append(f"## 待判斷 cluster（Top {top_k}，依 meme_quality 排序）\n")

    valid = comments[comments["cluster_label"] != -1]

    for _, row in analysis.head(top_k).iterrows():
        cid = int(row["cluster_id"])
        grp = valid[valid["cluster_label"] == cid]
        top5 = grp.nlargest(5, "like_count")[["text_clean", "like_count"]]

        lines.append(f"### Cluster {cid}")
        lines.append(f"- **size**：{row['size']} 則  **總讚數**：{row['likes_total']}  **最高讚**：{row['likes_max']}")
        lines.append(f"- **meme_quality**：{row['meme_quality']:.3f} "
                     f"(engage={row['engage_n']:.2f}, kw={row['kw_n']:.2f}, "
                     f"scene={row['scene_n']:.2f}, var={row['var_n']:.2f})")
        lines.append(f"- **主要場景**：{row['main_scene']} (sim={row['main_scene_sim']:.3f})")
        lines.append(f"- **關鍵字**：{row['keywords']}")
        lines.append(f"- **Top 5 留言**：")
        for i, r in enumerate(top5.itertuples(index=False), 1):
            text = str(r.text_clean)[:150].replace("\n", " ")
            lines.append(f"  {i}. [{int(r.like_count):>5}讚] {text}")
        lines.append("")

    return "\n".join(lines)


def render_noise(clusters_df: pd.DataFrame, n: int) -> str:
    """渲染 noise 中高讚留言（潛力候選）。"""
    noise = clusters_df[clusters_df["cluster_label"] == -1].nlargest(n, "like_count")
    if len(noise) == 0:
        return ""

    lines = ["## Noise 中的高讚孤狼（潛力迷因候選來源）\n"]
    lines.append(f"以下是未被分入任何 cluster 但讚數高的留言，可能是「尚未形成規模的新梗」。共 {n} 則：\n")
    for i, r in enumerate(noise.itertuples(index=False), 1):
        text = str(r.text_clean)[:200].replace("\n", " ")
        lines.append(f"{i}. [{int(r.like_count):>5}讚] {text}")
    lines.append("")
    return "\n".join(lines)


INVENTORY_TEMPLATE = """\
# meme_inventory_template.yaml — 輸出格式範本
# 請勿直接修改此檔案，填入內容後存為 meme_inventory_N.yaml

meta:
  source_file: review_for_judge.md
  total_clusters_judged: ~
  total_noise_candidates: ~

discovered_memes:
  - id: dm001
    cluster_id: ~
    name: ~
    type: ~
    canonical_text: ~
    canonical_likes: ~
    business_value: HIGH/MEDIUM/LOW
    marketing_angle: ~
    reasoning: ~
    matched_reference: ~

non_meme_clusters:
  - cluster_id: ~
    theme: ~
    why_not_meme: ~

potential_candidates:
  - source: noise
    text: ~
    likes: ~
    type: ~
    potential_value: HIGH/MEDIUM/LOW
    reasoning: ~
"""


def get_judge_prompt(inventory_path: Path) -> str:
    return f"""\
---

## 給 LLM Judge 的任務 Prompt

你（agent）的任務：根據上面提供的迷因風格示範，在待判斷的 cluster 與 noise 留言中
**找出新的迷因**，產出 `{inventory_path.name}`。

few-shot 範例是「人類認定的迷因長什麼樣」的示範，不是必須對應的答案；
若某個 cluster 剛好對應到某個範例，在 `matched_reference` 欄位註明，對不上就 null。

### 判斷規則

對於每個 cluster：
1. **讀 Top 5 留言、關鍵字、場景**，理解這群留言在討論什麼
2. **判斷是否為迷因**：
   - **真迷因**：有固定句式 / 角色化標籤 / 道具替代 / 引用台詞 / 互動格式
   - **非迷因**：只是觀眾在表達感想、討論劇情、評論演員
3. **如果是迷因**，分類到以下類型之一：
   - `catchphrase_chain`（接龍型，年份/次數 + 固定句式）
   - `format_template`（X就害誰 之類的格式仿作）
   - `character_meme`（圍繞特定角色的標籤化）
   - `quote_modification`（經典台詞二創改編）
   - `prop_joke`（道具/物件玩笑）
   - `trivia`（小知識/考據型）
   - `interactive`（接龍互動「以下是你...的次數 ⬇️」）

對於 noise 中的潛力候選：
- 用同樣的標準判斷它「是不是有潛力成為迷因」
- 考慮：單則讚數高 + 格式特殊 + 可被仿作 → 潛力高
- 純情感表達 → 不算潛力梗

### 商業價值評估

每個確認迷因給一個 marketing_angle：
- **HIGH**：可直接套用品牌行銷（情境共鳴強、格式可重用）
- **MEDIUM**：適合特定產業（食品、職場、保險...）
- **LOW**：負面標籤 / 過於小眾 / 考據型不適合 reuse

### 輸出格式

格式參考 `output/meme_inventory_template.yaml`，填入內容後存為 `{inventory_path.name}`：

```yaml
meta:
  source_file: review_for_judge.md
  total_clusters_judged: <K>
  total_noise_candidates: <N>

discovered_memes:         # 從 cluster 中辨識出的迷因（含對應到 few-shot 範例與全新發現）
  - id: dm001
    cluster_id: <id>
    name: <短名稱>
    type: <類型>
    canonical_text: <代表留言>
    canonical_likes: <讚數>
    business_value: HIGH/MEDIUM/LOW
    marketing_angle: <行銷方向>
    reasoning: <為什麼判定為迷因>
    matched_reference: <few-shot 中的 id，例如 m001；對不上填 null>

non_meme_clusters:        # 看似 top 但實為非迷因
  - cluster_id: <id>
    theme: <在討論什麼>
    why_not_meme: <為什麼不算迷因>

potential_candidates:     # noise 中的潛力梗
  - source: noise
    text: <留言原文>
    likes: <讚數>
    type: <推測類型>
    potential_value: HIGH/MEDIUM/LOW
    reasoning: <為什麼有潛力>
```

開始判斷吧。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"判斷的 cluster 數量（預設 {DEFAULT_TOP_K}）")
    parser.add_argument("--noise-n", type=int, default=NOISE_TOP_N,
                        help=f"noise 候選數量（預設 {NOISE_TOP_N}）")
    args = parser.parse_args()

    print("── Load ──")
    analysis = pd.read_parquet(DATA_DIR / "cluster_analysis.parquet")
    clusters_df = load_clusters()
    gt = load_ground_truth()
    print(f"  {len(analysis)} clusters, {len(clusters_df)} comments")

    print("\n── Render review ──")
    parts = []

    parts.append("# Meme Discovery — LLM Judge Review\n")
    parts.append(f"產生時間：依照 cluster.py + analyze.py 的最新結果\n")
    parts.append(f"待判斷 cluster：Top **{args.top_k}**（依 meme_quality 排序）")
    parts.append(f"Noise 候選：Top **{args.noise_n}**（依讚數）")
    parts.append("\n---\n")

    # 計算下一個 inventory 序號（meme_inventory_1.yaml, _2.yaml, ...）
    existing = list(OUTPUT_DIR.glob("meme_inventory_*.yaml"))
    next_n = max(
        (int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()),
        default=0,
    ) + 1
    inventory_path = OUTPUT_DIR / f"meme_inventory_{next_n}.yaml"

    # 寫 template 供 agent 參考格式
    template_path = OUTPUT_DIR / "meme_inventory_template.yaml"
    template_path.write_text(INVENTORY_TEMPLATE, encoding="utf-8")
    print(f"  Template → {template_path.name}")

    parts.append(render_few_shot(gt))
    parts.append(render_clusters(analysis, clusters_df, args.top_k))
    parts.append(render_noise(clusters_df, args.noise_n))
    parts.append(get_judge_prompt(inventory_path))

    out_path = OUTPUT_DIR / "review_for_judge.md"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"  Saved → {out_path}")
    print(f"\n下一步：開 Claude Code，請 agent 讀 {out_path.name}，"
          f"按檔案末尾的 prompt 寫 {inventory_path.name}")


if __name__ == "__main__":
    main()
