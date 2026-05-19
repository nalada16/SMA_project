# Final Project — 《甄嬛傳》迷因發掘系統

從 YouTube 留言中自動找出迷因、對應劇情、評估商業價值。

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
   ┌────────────┴────────────┬─────────────────┐
   ↓                          ↓                  ↓
analyze.py        classify_quote_vs_remix   scene_heatmap
   ↓                          ↓                  ↓
cluster_analysis  quote_classification    scene_heatmap.csv/png
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

會依序執行 cluster → analyze → quote → heatmap → judge，全部用 CPU 即可。

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
| `review_for_judge.md` | 給 Claude Code 判斷用 | Agent |
| `meme_inventory.yaml` | **最終迷因清單**（agent 寫）| 報告 |

---

## 關鍵指標說明

系統用以下幾個量化指標來找迷因、排序、分類。理解這幾個指標就能讀懂所有輸出。

### `meme_quality`（cluster 層級主分數，0–1）

每個 cluster 一個分數，越高代表「越像迷因」。由四個獨立訊號加權合成：

```
meme_quality = 0.40 × engagement_density   ← 讚數密度
             + 0.30 × kw_distinctiveness    ← 關鍵字鑑別度
             + 0.20 × scene_specificity     ← 場景對應強度
             + 0.10 × variation_richness    ← 內部變奏
```

四個訊號分別量什麼：

| 訊號 | 計算 | 反映什麼 | 例子 |
|---|---|---|---|
| `engagement_density` | `log(likes_total / size + 1)` | 平均每則留言能拿多少讚 | 小群高讚（如「果子狸」36 則 9450 讚）得分高 |
| `kw_distinctiveness` | c-TF-IDF top-3 字元 n-gram 的 IDF 平均 | 是否有獨特 catchphrase | 「巧克力球」「果子狸」這種獨佔詞高，「甄嬛」「皇上」這種大家都用的低 |
| `scene_specificity` | 群內留言對最近場景的 cosine sim 平均 | 是否指向具體劇情點 | 「熹妃回宮」明確指向 ep56-891，分數高；空泛感想低 |
| `variation_richness` | 群內留言 embedding 對 centroid 的距離標準差 | 群內是不是有「變奏」 | 接龍年份不同、填空換主角 → 標準差大 → 真的在玩這個梗 |

> 四個訊號**各自在這次分析的所有 cluster 內做 min-max normalize 到 0–1** 後再加權。所以 `meme_quality` 是**相對排名**，數值在不同資料集間不能直接比較。

---

### Quote vs Remix 三分類（cluster 層級）

每個 cluster 對應到劇本最相似的台詞，依該 sim 分三類：

| 類型 | sim 範圍 | 意義 | 例子 |
|---|---|---|---|
| `direct_quote` | sim > 0.75 | 幾乎是原文照搬 | 留言「太監：熹妃回宮——」直接引用 |
| `template_modification` | 0.5 < sim ≤ 0.75 | 套格式仿作 | 「X：我就害誰」格式換主詞 |
| `creative_derivative` | sim ≤ 0.5 | 二次創作，脫離原台詞 | 改編、玩梗、新比喻 |

每群輸出 `avg_top_sim`（前 5 高讚留言對最近台詞的平均 sim）作為分類依據。

---

### 場景熱度指標（scene 層級）

每個劇情場景（ep56-894 等）依以下排序：

| 指標 | 意義 |
|---|---|
| `n_clusters` | 多少個 cluster 以這幕為主場景 |
| `total_likes` | 對應 cluster 的總讚數加總（最直接的「熱度」指標） |
| `avg_meme_qual` | 對應 cluster 的平均 meme_quality |
| `avg_scene_sim` | 留言對此幕的平均 cosine sim |
| `top_clusters` | 該幕對應的 Top-3 cluster ID + 讚數 |

---

### 角色排行指標（character 層級）

每個角色（皇帝、甄嬛、安陵容...）的台詞被多少 cluster 引用：

| 指標 | 意義 |
|---|---|
| `被迷因化次數` | 多少個 cluster 對應到該角色的台詞 |
| `總讚數` | 這些 cluster 的讚數加總 |
| `平均每群讚數` | 「每出場一次能爆多少」（影響力效率） |
| `平均sim` | 對應台詞的平均 quote sim |

---

### Cluster 基本欄位

| 欄位 | 意義 |
|---|---|
| `cluster_label` | 群編號，`-1` 表示 noise（沒被歸進任何群） |
| `size` | 群內留言數 |
| `likes_total` / `likes_max` | 群內總讚數 / 最高讚的留言 |
| `is_interactive_candidate` | 是否為 noise 中讚數 ≥ 500 的「潛力梗候選」 |

---

### Pipeline 過濾門檻

| 參數 | 預設值 | 意義 |
|---|---|---|
| `F_MIN_LIKES` | 2 | 留言至少要有 2 讚才進分群（過濾雜訊） |
| `INTERACTIVE_MIN_LIKES` | 500 | noise 留言讚數 ≥ 此值列為潛力候選 |
| `MIN_CLUSTER_SIZE` | 20 | HDBSCAN 一群至少 20 則留言 |

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
