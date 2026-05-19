"""
embed_scripts.py
解析《甄嬛傳》劇本台詞，按「幕」切分後用 Qwen3-Embedding-8B 做向量嵌入。
輸出：SMA/data/scripts_with_embedding.parquet

VRAM 需求：
    FP16（預設）→ 模型本體 16 GB，需要 16GB+ VRAM
    INT4 量化   → 模型本體  4 GB，8GB VRAM 可用（加 --quantize）

Usage：
    uv run SMA/embed_scripts.py --dry-run           # 只解析，測試用
    uv run SMA/embed_scripts.py --quantize          # 8GB VRAM 用這個
    uv run SMA/embed_scripts.py --quantize --batch-size 2
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
OUTPUT_PATH = DATA_DIR / "scripts_with_embedding.parquet"
MAX_LENGTH  = 512

EPISODE_FILES = {
    "ep56": (56, SCRIPT_DIR / "ep56.txt"),
    "ep63": (63, SCRIPT_DIR / "ep63.txt"),
    "ep76": (76, SCRIPT_DIR / "ep76.txt"),
}

SCENE_RE = re.compile(r'第(\d+)幕(（续）)?')


def parse_episode(episode_key: str, episode_num: int, filepath: Path) -> list[dict]:
    raw    = filepath.read_text(encoding="utf-8")
    splits = list(SCENE_RE.finditer(raw))
    if not splits:
        print(f"  WARNING: no scenes found in {filepath.name}")
        return []
    scenes = []
    for i, match in enumerate(splits):
        scene_num  = int(match.group(1))
        is_cont    = match.group(2) is not None
        start      = match.end()
        end        = splits[i + 1].start() if i + 1 < len(splits) else len(raw)
        scene_text = raw[start:end].strip()
        if scene_text:
            scenes.append({
                "episode":         episode_key,
                "episode_num":     episode_num,
                "scene_number":    scene_num,
                "is_continuation": is_cont,
                "text":            scene_text,
            })
    return scenes


def last_token_pool(last_hidden_states, attention_mask):
    import torch
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def load_model(quantize: bool):
    """
    載入模型。
    quantize=True  → INT4（bitsandbytes），適合 8GB VRAM
    quantize=False → FP16，需要 16GB+ VRAM
    """
    import torch
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    if device == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM   : {total_vram:.1f} GB")
        if total_vram < 12 and not quantize:
            print("WARNING: VRAM < 12GB，FP16 可能 OOM，建議加上 --quantize")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if quantize:
        print("Loading model in INT4 (bitsandbytes) ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        ).eval()
    else:
        print("Loading model in FP16 ...")
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
        ).to(device).eval()

    return tokenizer, model, device


def embed_scenes(scenes: list[dict], batch_size: int, quantize: bool) -> list[dict]:
    import torch
    import torch.nn.functional as F

    tokenizer, model, device = load_model(quantize)
    texts = [s["text"] for s in scenes]
    all_embeddings = []

    print(f"Embedding {len(scenes)} scenes (batch_size={batch_size}) ...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch   = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        embs = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
        embs = F.normalize(embs, p=2, dim=1)
        all_embeddings.extend(embs.cpu().float().numpy())

    for scene, emb in zip(scenes, all_embeddings):
        scene["embedding"] = emb
    return scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true",
                        help="只解析，不做 embedding")
    parser.add_argument("--quantize",  action="store_true",
                        help="INT4 量化載入（8GB VRAM 必須加這個）")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size（INT4 建議 2–4，FP16 可用 8）")
    args = parser.parse_args()

    print("── Parsing ───────────────────────────────────────────────")
    all_scenes = []
    for key, (num, path) in EPISODE_FILES.items():
        if not path.exists():
            print(f"  SKIP: {path} not found")
            continue
        scenes = parse_episode(key, num, path)
        print(f"  {key}: {len(scenes)} scenes")
        all_scenes.extend(scenes)
    print(f"  Total: {len(all_scenes)} scenes\n")

    if not all_scenes:
        print("No scenes found.")
        return

    if args.dry_run:
        df = pd.DataFrame(all_scenes)
        pd.set_option("display.max_colwidth", 80)
        print(df[["episode", "scene_number", "is_continuation", "text"]].to_string())
        print(f"\n{len(df)} scenes parsed. (dry-run: no embedding)")
        return

    print("── Embedding ─────────────────────────────────────────────")
    all_scenes = embed_scenes(all_scenes, batch_size=args.batch_size, quantize=args.quantize)

    print("\n── Saving ────────────────────────────────────────────────")
    df = pd.DataFrame(all_scenes)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}  ({len(df)} scenes)")


if __name__ == "__main__":
    main()
