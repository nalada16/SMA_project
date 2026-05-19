"""
embed_lines.py
解析劇本，把每一句對白單獨切出來、做 line-level Qwen3 embedding。

格式假設：
    第891幕                          ← scene marker
    （皇帝皇后...）                  ← stage direction (跳過)
    太監：熹妃回宮——                ← dialogue line（角色 + 「：」 + 對白）

輸出：data/lines_with_embedding.parquet
    欄位：line_id, episode, scene_number, character, text, line_index, embedding

Usage：
    uv run final_project/embed_lines.py --dry-run
    uv run final_project/embed_lines.py --quantize --batch-size 8
"""
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
SCRIPT_DIR  = ROOT / "huan_scripts"
MODEL_NAME  = "Qwen/Qwen3-Embedding-8B"
OUTPUT_PATH = DATA_DIR / "lines_with_embedding.parquet"
MAX_LENGTH  = 256  # 對白通常很短，256 已綽綽有餘

EPISODE_FILES = {
    "ep56": (56, SCRIPT_DIR / "ep56.txt"),
    "ep63": (63, SCRIPT_DIR / "ep63.txt"),
    "ep76": (76, SCRIPT_DIR / "ep76.txt"),
}

SCENE_RE = re.compile(r'第(\d+)幕(（续）)?')
# 對白格式：「角色名：對白內容」
# 角色名 1–6 個中文字，全形冒號或半形冒號都接受
LINE_RE = re.compile(r'^([一-鿿]{1,6})[:：](.+)$')


def parse_episode(episode_key: str, episode_num: int, filepath: Path) -> list[dict]:
    """把劇本拆成 line-level，每一句對白一筆。"""
    raw_lines = filepath.read_text(encoding="utf-8").splitlines()
    rows = []

    current_scene = None
    line_index_in_scene = 0

    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue

        # 場景標記
        m = SCENE_RE.match(line)
        if m:
            current_scene = int(m.group(1))
            line_index_in_scene = 0
            continue

        # 跳過舞台指示（括號內容）
        if line.startswith("（") or line.startswith("("):
            continue

        # 對白
        m = LINE_RE.match(line)
        if not m or current_scene is None:
            continue

        character = m.group(1).strip()
        text = m.group(2).strip()
        if not text:
            continue

        rows.append({
            "line_id":      f"{episode_key}-{current_scene}-{line_index_in_scene:03d}",
            "episode":      episode_key,
            "episode_num":  episode_num,
            "scene_number": current_scene,
            "character":    character,
            "text":         text,
            "line_index":   line_index_in_scene,
        })
        line_index_in_scene += 1

    return rows


def last_token_pool(last_hidden_states, attention_mask):
    import torch
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device),
        seq_lens,
    ]


def load_model(quantize: bool):
    import torch
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM   : {vram:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if quantize:
        print("Loading model in INT4 ...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModel.from_pretrained(
            MODEL_NAME, quantization_config=bnb, device_map="auto"
        ).eval()
    else:
        print("Loading model in FP16 ...")
        model = AutoModel.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        ).to(device).eval()

    return tokenizer, model, device


def embed_lines(rows: list[dict], batch_size: int, quantize: bool) -> list[dict]:
    import torch
    import torch.nn.functional as F

    tokenizer, model, device = load_model(quantize)
    texts = [r["text"] for r in rows]
    embeddings = []

    print(f"Embedding {len(rows)} lines (batch_size={batch_size}) ...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        embs = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
        embs = F.normalize(embs, p=2, dim=1)
        embeddings.extend(embs.cpu().float().numpy())

    for r, e in zip(rows, embeddings):
        r["embedding"] = e
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="只解析，不做 embedding")
    parser.add_argument("--quantize", action="store_true",
                        help="INT4 量化（8GB VRAM 必須加）")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="lines 比 scenes 短，batch 可以大一些")
    args = parser.parse_args()

    print("── Parsing ──")
    all_rows = []
    for key, (num, path) in EPISODE_FILES.items():
        if not path.exists():
            print(f"  SKIP: {path} not found")
            continue
        rows = parse_episode(key, num, path)
        print(f"  {key}: {len(rows)} lines")
        all_rows.extend(rows)
    print(f"  Total: {len(all_rows)} lines\n")

    if not all_rows:
        print("No lines found.")
        return

    if args.dry_run:
        df = pd.DataFrame(all_rows)
        pd.set_option("display.max_colwidth", 60)
        print(df[["line_id", "character", "text"]].head(30).to_string())
        print(f"\n{len(df)} lines parsed. Top characters by line count:")
        print(df["character"].value_counts().head(10))
        return

    print("── Embedding ──")
    all_rows = embed_lines(all_rows, batch_size=args.batch_size, quantize=args.quantize)

    print("\n── Saving ──")
    df = pd.DataFrame(all_rows)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}  ({len(df)} lines)")


if __name__ == "__main__":
    main()
