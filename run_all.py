"""
run_all.py — 一鍵跑完整 pipeline。

順序：
    1. cluster.py                   分群
    2. analyze.py                   場景對應 + 關鍵字 + meme_quality
    3. classify_quote_vs_remix.py   引用 vs 改編
    4. scene_heatmap.py             場景熱度排行
    5. meme_potential.py            潛力預測（需 lines_with_embedding.parquet）
    6. judge.py                     產生 review_for_judge.md（給 Claude Code 讀）

不會自動跑：
    embed_scripts.py / embed_lines.py — 需要 GPU，請手動執行
    judge 階段最後一步 — agent 讀 review.md 寫 inventory.yaml

Usage:
    uv run final_project/run_all.py
    uv run final_project/run_all.py --skip-potential  # 跳過 meme_potential
"""
import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent


STEPS = [
    ("cluster",           "cluster.py",                "分群 (UMAP + HDBSCAN)"),
    ("analyze",           "analyze.py",                "場景對應 + 關鍵字 + meme_quality"),
    ("quote",             "classify_quote_vs_remix.py","引用 vs 改編分類"),
    ("heatmap",           "scene_heatmap.py",          "場景熱度排行 + heatmap"),
    ("potential",         "meme_potential.py",         "迷因潛力預測"),
    ("judge",             "judge.py",                  "產生 LLM judge review.md"),
]


def run_step(script_name: str) -> bool:
    script = ROOT / script_name
    print(f"\n{'='*70}")
    print(f"執行：{script_name}")
    print('='*70)
    result = subprocess.run(
        ["uv", "run", "python", "-X", "utf8", str(script)],
        cwd=ROOT.parent,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-potential", action="store_true",
                        help="跳過 meme_potential（如果 lines_with_embedding.parquet 未生成）")
    parser.add_argument("--only", nargs="+", choices=[s[0] for s in STEPS],
                        help="只跑指定的 step")
    args = parser.parse_args()

    # 預先檢查必要資料
    data_dir = ROOT / "data"
    required = ["comments_with_embedding.parquet", "scripts_with_embedding.parquet"]
    for f in required:
        if not (data_dir / f).exists():
            print(f"❌ 缺少必要資料：{f}")
            print(f"   請先放在 {data_dir}/")
            sys.exit(1)

    lines_path = data_dir / "lines_with_embedding.parquet"
    if not lines_path.exists() and not args.skip_potential:
        print(f"⚠️  {lines_path.name} 不存在")
        print(f"   meme_potential 和 classify_quote_vs_remix 需要這個檔案")
        print(f"   請先跑：uv run final_project/embed_lines.py --quantize")
        print(f"   或加上 --skip-potential 跳過")
        sys.exit(1)

    steps_to_run = STEPS
    if args.only:
        steps_to_run = [s for s in STEPS if s[0] in args.only]
    if args.skip_potential:
        steps_to_run = [s for s in steps_to_run if s[0] not in ("potential", "quote")]

    print(f"預計執行 {len(steps_to_run)} 個步驟：")
    for key, script, desc in steps_to_run:
        print(f"  - {key}: {desc}")

    failed = []
    for key, script, desc in steps_to_run:
        ok = run_step(script)
        if not ok:
            failed.append(key)
            print(f"\n❌ {key} 失敗")
            if input("繼續？(y/N): ").lower() != "y":
                break

    print(f"\n{'='*70}")
    if failed:
        print(f"完成，{len(failed)} 個步驟失敗：{failed}")
    else:
        print("全部完成！")
        print(f"\n下一步：開 Claude Code，請 agent 讀")
        print(f"  final_project/output/review_for_judge.md")
        print(f"並依照檔案末尾的 prompt 寫")
        print(f"  final_project/output/meme_inventory.yaml")


if __name__ == "__main__":
    main()
