# Final Project — 《甄嬛傳》迷因發掘系統

從 YouTube 留言中自動找出迷因、對應劇情、評估商業價值，並產出潛力梗預測。

> 這個資料夾是 **單次跑、不含 autoresearch 實驗迴圈** 的完整版本。所有參數已從 autoresearch 找到的最佳組合固定下來。

---

## 系統流程

```
劇本 .txt              留言（含 4096 維 Qwen3 embedding）
   ↓                          ↓
embed_scripts.py        embed_lines.py
   ↓                          ↓
場景 embedding (27)    對白行 embedding (577)
   ↓                          ↓
            cluster.py
                ↓
        clusters.parquet (86 群 + noise)
                ↓
   ┌────────────┴────────────┬─────────────────┬───────────────┐
   ↓                          ↓                  ↓                ↓
analyze.py        classify_quote_vs_remix   scene_heatmap   meme_potential
   ↓                          ↓                  ↓                ↓
cluster_analysis  quote_classification    scene_heatmap     potential_predictions
                                              ↓
                                          judge.py → review_for_judge.md
                                              ↓
                                  （由 Claude Code agent 讀完寫）
                                              ↓
                                       meme_inventory.yaml
```

---

## 檔案說明

### Scripts（按執行順序）

| 檔案 | 功能 | 需先跑 |
|---|---|---|
| `embed_scripts.py` | 把 EP56/63/76 劇本切成 27 個「幕」做 embedding | — |
| `embed_lines.py` | 把劇本切成 577 句對白做 line-level embedding | — |
| `cluster.py` | 用最佳 pipeline 跑分群（UMAP + HDBSCAN）| comments_with_embedding.parquet |
| `analyze.py` | 場景對應 + c-TF-IDF + meme_quality 評分 | cluster.py |
| `classify_quote_vs_remix.py` | 把每群分類成 引用 / 改編 / 二創 | cluster.py + embed_lines.py |
| `scene_heatmap.py` | 場景引爆熱度排行 + heatmap 視覺化 | analyze.py |
| `meme_potential.py` | 從台詞特徵預測潛力梗 | embed_lines.py |
| `judge.py` | 產生 LLM judge 用的 review.md（給 agent 讀）| analyze.py |
| `run_all.py` | 一鍵跑除了 embed 之外的所有步驟 | — |

### 共用工具

| 檔案 | 功能 |
|---|---|
| `prepare.py` | 資料載入函式（comments / scenes / lines / clusters / ground_truth）|
| `ground_truth.yaml` | 8 個已驗證迷因 + 1 個潛力候選（給 judge.py 當 few-shot 範例）|
| `explore_results.ipynb` | 互動瀏覽 notebook |

### 資料夾

```
final_project/
├── huan_scripts/           劇本原文 .txt
├── data/                   parquet / npy 資料（輸入 + 中間產物）
└── output/                 最終分析輸出（給人讀的 csv / png / yaml）
```

---

## 第一次執行流程

### 1. 環境準備

```bash
uv sync
uv pip install accelerate bitsandbytes  # GPU 推論需要
```

### 2. 產生 embedding（**需 GPU**，8GB VRAM 加 --quantize 即可）

```bash
uv run final_project/embed_scripts.py --quantize     # 27 場景，~10 分鐘
uv run final_project/embed_lines.py --quantize       # 577 對白，~15 分鐘
```

### 3. 跑分群和分析

```bash
uv run final_project/run_all.py
```

會依序執行 cluster → analyze → quote → heatmap → potential → judge，全部用 CPU 即可。

### 4. 最後一步：LLM judge（半自動）

開 Claude Code，請 agent 讀 `output/review_for_judge.md`，照檔案末尾的 prompt 寫 `output/meme_inventory.yaml`。

review.md 已經包含：
- 8 個 ground truth 迷因作為 **few-shot 範例**
- Top 20 cluster 的留言、關鍵字、場景對應
- Top 20 noise 候選
- 完整的判斷 prompt

---

## 輸出檔案

### data/ （中間產物）

| 檔案 | 內容 |
|---|---|
| `lines_with_embedding.parquet` | 577 句對白 + 4096 維 embedding |
| `scripts_with_embedding.parquet` | 27 個場景 + embedding |
| `clusters.parquet` | 每則留言 + cluster_label + umap 座標 |
| `umap_50d_cache.npy` | UMAP 降維結果快取 |
| `cluster_analysis.parquet` | 每群的 size / likes / 關鍵字 / 場景 / meme_quality |
| `comment_scene_mapping.parquet` | 每則留言對應到最近的場景 |

### output/ （最終分析）

| 檔案 | 內容 | 給誰看 |
|---|---|---|
| `quote_classification.csv` | 每群分類為 direct_quote / template / creative | 報告 |
| `scene_heatmap.csv` | 場景引爆熱度排行 | 報告 |
| `scene_heatmap.png` | 視覺化 heatmap | 報告 |
| `potential_predictions.csv` | 所有台詞的迷因潛力分數 | 報告 |
| `review_for_judge.md` | 給 Claude Code 判斷用 | Agent |
| `meme_inventory.yaml` | **最終迷因清單**（agent 寫）| 報告 |

---

## 最佳 Pipeline 參數（已固定）

來自 autoresearch 實驗找到的最佳組合：

| 階段 | 設定 |
|---|---|
| Prefilter | `like_count ≥ 2` |
| Embedding | Qwen3-Embedding-8B 4096 維 |
| Dimensionality Reduction | UMAP n_components=50, n_neighbors=15, min_dist=0.0, cosine metric |
| Clustering | HDBSCAN min_cluster_size=20, min_samples=1, eom |
| Postprocess | Interactive detector（noise + likes ≥ 500）|
| Meme Scoring | engagement 0.40 + kw 0.30 + scene 0.20 + var 0.10 |

要試其他組合，請看 autoresearch 版本（`SMA/experiment/`）。

---

## 換集數的使用方式

如果想跑其他集數（EP1, EP30, ...）：

1. 在 `huan_scripts/` 放新集數的 `epXX.txt`
2. 修改 `embed_scripts.py` 和 `embed_lines.py` 裡的 `EPISODE_FILES` 加上新集數
3. 替換 `data/comments_with_embedding.parquet` 為新集數的留言資料（格式：含 `text_clean`, `like_count`, `keep`, `embedding` 4096 維）
4. 跑 `embed_scripts.py`、`embed_lines.py`、`run_all.py`

ground_truth.yaml 也要對應更新（這集的已知迷因），不然 judge 的 few-shot 會錯位。
