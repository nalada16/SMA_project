"""
meme_potential.py — 從台詞特徵預測迷因潛力

做法：
    1. 從 ground_truth 取出已知迷因對應的台詞，當作「高潛力」正樣本
    2. 從 lines_with_embedding.parquet 抽特徵：
       - 句長
       - 角色
       - 標點密度
       - 是否含指代詞（我、你、朕、本宮）
       - 是否含情緒/特殊用詞
    3. 用 LogisticRegression 訓練二元分類器，預測「會不會成為迷因」
    4. 對所有未標記台詞打分，找出「結構像迷因但還沒爆」的潛力句

需要先跑：
    embed_lines.py

Output:
    output/potential_predictions.csv

Usage:
    uv run final_project/meme_potential.py
"""
import sys
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_lines, load_ground_truth, OUTPUT_DIR

# 簡單的指代詞、情緒詞表
PRONOUNS = ["我", "你", "朕", "本宮", "臣妾", "妾身", "奴婢", "本王"]
EMOTION_WORDS = ["不過", "竟然", "難道", "怎麼", "豈", "也罷",
                 "罷了", "便是", "倒也", "原來", "可笑", "好得很"]
ENDING_PARTICLES = ["啊", "呢", "嗎", "吧", "呀", "哎", "了", "麼"]


def extract_features(text: str, character: str) -> dict:
    """從單句台詞抽特徵"""
    return {
        "length":        len(text),
        "is_short":      int(len(text) <= 10),
        "is_medium":     int(10 < len(text) <= 25),
        "is_long":       int(len(text) > 25),
        "n_pronouns":    sum(text.count(p) for p in PRONOUNS),
        "n_emotion":     sum(int(w in text) for w in EMOTION_WORDS),
        "n_particles":   sum(text.count(p) for p in ENDING_PARTICLES),
        "n_punct":       sum(text.count(p) for p in "，。？！、；："),
        "has_question":  int("？" in text or "嗎" in text),
        "has_exclaim":   int("！" in text),
        "is_huang":      int(character in ["皇帝", "皇上"]),
        "is_zhen":       int(character == "甄嬛"),
        "is_anlin":      int(character in ["陵容", "安陵容"]),
        "is_huangfei":   int(character == "皇后"),
        "is_eunuch":     int(character in ["蘇培盛", "小允子", "太監"]),
        "is_other":      int(character not in ["皇帝", "皇上", "甄嬛", "陵容", "安陵容",
                                                "皇后", "蘇培盛", "小允子", "太監"]),
    }


def find_gt_meme_lines(lines: pd.DataFrame, gt: dict) -> set:
    """
    對每個 ground truth 迷因，找出最相似的台詞（用其 canonical_text）作為正樣本標記。
    回傳 line_id 集合。
    """
    line_emb = np.stack(lines["embedding"].values)
    line_emb = line_emb / (np.linalg.norm(line_emb, axis=1, keepdims=True) + 1e-9)

    positives = set()

    print("\n── 標記正樣本（已知迷因的對應台詞）──")
    for meme in gt["confirmed_memes"]:
        scene_ref = meme.get("scene_ref")
        keywords = meme.get("signature_keywords", [])
        if not keywords:
            continue

        kw_pattern = "|".join([re.escape(k) for k in keywords])
        candidates = lines[lines["text"].str.contains(kw_pattern, regex=True, na=False)]

        if scene_ref:
            # 限制在該場景內
            cand_in_scene = candidates[
                candidates.apply(
                    lambda r: f"{r['episode']}-{r['scene_number']}" == scene_ref,
                    axis=1
                )
            ]
            if len(cand_in_scene) > 0:
                candidates = cand_in_scene

        if len(candidates) == 0:
            print(f"  {meme['id']} ({meme['name']}): 找不到匹配台詞")
            continue

        for _, r in candidates.iterrows():
            positives.add(r["line_id"])
        print(f"  {meme['id']} ({meme['name']}): {len(candidates)} 條台詞標為正樣本")

    return positives


def main():
    print("── Load ──")
    lines = load_lines()
    gt = load_ground_truth()
    print(f"  {len(lines)} lines")

    # 1. 抽特徵
    print("\n── 抽特徵 ──")
    feats = lines.apply(
        lambda r: pd.Series(extract_features(r["text"], r["character"])),
        axis=1
    )
    feat_cols = feats.columns.tolist()
    df = pd.concat([lines.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    print(f"  features: {feat_cols}")

    # 2. 標記正樣本
    positives = find_gt_meme_lines(lines, gt)
    df["is_meme"] = df["line_id"].isin(positives).astype(int)
    n_pos = int(df["is_meme"].sum())
    print(f"\n  正樣本: {n_pos} / 總台詞 {len(df)}")

    if n_pos < 3:
        print("  正樣本太少，無法訓練分類器")
        return

    # 3. 訓練 LogisticRegression
    print("\n── 訓練分類器 ──")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = df[feat_cols].values.astype(float)
    y = df["is_meme"].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_s, y)

    # In-sample 報告（樣本太少，不做 cross-validation）
    proba = clf.predict_proba(X_s)[:, 1]
    df["potential_score"] = proba

    # 特徵重要性（用係數絕對值大小）
    coefs = sorted(zip(feat_cols, clf.coef_[0]), key=lambda x: -abs(x[1]))
    print("  Top 5 特徵權重：")
    for name, val in coefs[:5]:
        sign = "+" if val > 0 else "-"
        print(f"    {sign} {name:<15}  {val:+.3f}")

    # 4. 找出「結構像迷因但還沒標記」的潛力句
    print("\n── Top 20 潛力台詞（未在 GT 內、predicted_score 最高）──")
    unlabeled = df[df["is_meme"] == 0].nlargest(20, "potential_score")

    for _, r in unlabeled.iterrows():
        print(f"  [{r['potential_score']:.3f}] {r['character']}：{r['text'][:60]}")
        print(f"          {r['episode']}-{r['scene_number']}")

    # 儲存
    cols_out = ["line_id", "episode", "scene_number", "character", "text",
                "is_meme", "potential_score"] + feat_cols
    out = df[cols_out].sort_values("potential_score", ascending=False).reset_index(drop=True)

    out_path = OUTPUT_DIR / "potential_predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
