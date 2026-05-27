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
                                       meme_inventory_N.yaml
```

---

## 檔案說明

### Scripts（按執行順序）

| 檔案 | 功能 | 需先跑 | 產出檔案 |
|---|---|---|---|
| `embed_scripts.py` | 把 EP56/63/76 劇本切成 27 個「幕」做 embedding | — | `data/scripts_with_embedding.parquet` |
| `embed_lines.py` | 把劇本切成 577 句對白做 line-level embedding | — | `data/lines_with_embedding.parquet` |
| `cluster.py` | 用最佳 pipeline 跑分群（UMAP + HDBSCAN）| comments_with_embedding.parquet | `data/clusters.parquet` `data/umap_50d_cache.npy` |
| `analyze.py` | 場景對應 + c-TF-IDF + meme_quality 評分 | cluster.py | `data/cluster_analysis.parquet` `data/comment_scene_mapping.parquet` |
| `classify_quote_vs_remix.py` | 把每群分類成 引用 / 改編 / 二創 | cluster.py + embed_lines.py | `output/quote_classification.csv` |
| `scene_heatmap.py` | 場景引爆熱度排行 + heatmap 視覺化 | analyze.py | `output/scene_heatmap.csv` `output/scene_heatmap.png` |
| `judge.py` | 產生 LLM judge 用的 review.md（給 agent 讀）| analyze.py | `output/review_for_judge.md` `output/meme_inventory_template.yaml` |
| `run_all.py` | 一鍵跑除了 embed 之外的所有步驟 | — | （同上各步驟）|

### 共用工具

| 檔案 | 功能 |
|---|---|
| `prepare.py` | 資料載入函式（comments / scenes / lines / clusters / ground_truth）|
| `ground_truth.yaml` | 25 個已知迷因 (m001–m025) + 1 個潛力候選（給 judge.py 當迷因風格示範，非答案鍵）|
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

開新的 Claude Code session，貼以下 prompt 啟動 agent：

```
你的任務是從甄嬛傳 YouTube 留言 cluster 中找出具商業價值的迷因。

第一步：讀這個檔案
final_project/output/review_for_judge.md

檔案包含：
1. Few-shot 範例（人類認定的迷因，作為風格參考）
2. 待判斷的 cluster 資料（Top 20，含留言、關鍵字、場景）
3. Noise 中的高讚孤狼留言
4. 檔案末尾有完整的判斷規則與輸出格式說明

限制：
- 不要讀任何 meme_inventory_*.yaml 舊檔案，避免先入為主
- few-shot 範例是風格示範，不是你必須找到的答案
- 輸出檔名已在 review_for_judge.md 末尾指定，直接用那個名字

讀完後依照末尾 prompt 開始判斷。
```

每次跑 `judge.py` 會自動計算下一個序號，review.md 末尾的 prompt 裡直接寫明目標檔名（`meme_inventory_1.yaml`、`meme_inventory_2.yaml`…），不需要手動改名。

review.md 已經包含：
- 25 個 ground truth 迷因作為**迷因風格示範**（風格參考，非答案鍵）
- Top 20 cluster 的留言、關鍵字、場景對應
- Top 20 noise 候選
- 完整的判斷 prompt + 輸出格式

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
| `meme_inventory_template.yaml` | 輸出格式範本（judge.py 每次重寫）| Agent |
| `meme_inventory_N.yaml` | **最終迷因清單**（agent 每次跑產出，序號自動遞增）| 報告 |

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

## 換資料 / 擴充集數

### 你需要準備的三件事

#### 1. 留言資料 `data/comments_with_embedding.parquet`

DataFrame 必須包含這幾個欄位（其他欄位會被忽略）：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `comment_id` | string | 留言唯一 ID（用來去重、合併用）|
| `text_clean` | string | 留言文字（已清理，去除 emoji 過多/換行/特殊符號可選）|
| `like_count` | int | 按讚數（用於 prefilter 和 ranking）|
| `keep` | bool | 是否保留分析（`False` 會被 `prepare.py:load_comments()` 過濾）|
| `embedding` | numpy.ndarray (4096,) | Qwen3-Embedding-8B 的向量，**已 L2 normalize** |

**embedding 重點**：
- 維度必須是 **4096**，跟劇本的 `scripts_with_embedding.parquet` 一致（同個模型才能 cosine similarity）
- 必須是 numpy array（parquet 存 list 也行，但要能用 `np.stack(df["embedding"].values)` 還原成 (N, 4096) 矩陣）
- 建議 normalize 後再存（雖然 cosine similarity 內部會重算，但 normalized embedding 跨檔案運算更安全）

最小驗證腳本：
```python
import pandas as pd
import numpy as np
df = pd.read_parquet("data/comments_with_embedding.parquet")
assert {"comment_id", "text_clean", "like_count", "keep", "embedding"} <= set(df.columns)
X = np.stack(df["embedding"].values)
assert X.shape[1] == 4096, f"embedding dim should be 4096, got {X.shape[1]}"
print(f"✓ {len(df)} comments, X.shape={X.shape}")
```

#### 2. 劇本原文 `huan_scripts/epXX.txt`

純文字檔，格式如下（每集一個檔）：

```
第891幕
（？皇帝皇后和众嫔妃迎候甄嬛）
太监：熹妃回宫——
甄嬛：臣妾归来，恭祝皇上、皇后圣体康健、福泽万年。
皇帝：一路可好吗？
...

第892幕
（永寿宫）
皇帝：朕知道你喜欢赏莲...
```

格式規則：
- **場景標記**：`第XXX幕` 或 `第XXX幕（续）`
- **舞台指示**：用全形括弧 `（...）` 包起來，會被自動跳過
- **對白行**：`角色：對白`（全形或半形冒號都可），角色名 1–6 個中文字
- 空行隨意，會自動忽略

完成後，到 `embed_scripts.py` 和 `embed_lines.py` 裡加上新集數：

```python
EPISODE_FILES = {
    "ep56": (56, SCRIPT_DIR / "ep56.txt"),
    "ep63": (63, SCRIPT_DIR / "ep63.txt"),
    "ep76": (76, SCRIPT_DIR / "ep76.txt"),
    "ep1":  (1,  SCRIPT_DIR / "ep1.txt"),   # 新增這行
}
```

#### 3. Ground Truth `ground_truth.yaml`

LLM judge 用這個檔案作為 few-shot 範例，所以**換集數的時候 ground truth 也要換**，不然會用其他集數的迷因當示範，判斷就會偏。

格式：

```yaml
confirmed_memes:
  - id: m001                                # 內部 ID，m001 / m002 ... 流水號
    name: 熹妃回宮接龍                       # 人類看得懂的名稱
    type: catchphrase_chain                  # 類型，見下方七選一
    signature_keywords: [熹妃, 回宮, 恭迎]    # 偵測用的關鍵字，會用 regex OR 比對
    canonical_text: "2023了 我還在恭迎熹妃娘娘回宮😚"   # 代表留言原文
    canonical_likes: 3707                    # 那則留言的讚數（不知道填 null）
    scene_ref: ep56-891                      # 對應的場景，格式 epXX-XXX（不知道填 null）
    algo_verified: true                      # 演算法是否能找到（首次標時設 false 也 OK）
    usage_note: "接龍型，每年都有人留言宣告自己還在看"  # 網友怎麼用這個梗（選填，但有助 LLM 判斷）

potential_memes:                             # 潛力梗（noise 中的）
  - id: p001
    name: 你看了幾次接龍
    type: interactive
    signature_keywords: [看過, 次數]
    canonical_text: "以下是你看過甄嬛傳的次數 ⬇️"
    canonical_likes: 14070
    scene_ref: null                          # 沒對應特定場景就 null
    discovery_signal: noise_high_like
```

**七種 type**（給 LLM 判斷時的分類選項）：

| type | 說明 | 例子 |
|---|---|---|
| `catchphrase_chain` | 接龍型，固定句式 + 變動部分（年份/次數）| 「2024 了我還在恭迎熹妃娘娘回宮」 |
| `format_template` | 套用格式仿作 | 「X：誰Y我就害誰」 |
| `character_meme` | 圍繞特定角色的標籤化 | 「小允子神隊友」「寧貴人風紀股長」 |
| `quote_modification` | 經典台詞二創改編 | 「果子狸→胖維尼」 |
| `prop_joke` | 道具/物件玩笑 | 「巧克力球」 |
| `trivia` | 小知識／考據型 | 「皇上拉劍帶冤死」 |
| `interactive` | 邀請互動的格式 | 「以下是你 X 過 Y 的次數 ⬇️」 |

**怎麼標 ground truth**：
1. 開 `explore_results.ipynb` 看 Top 15 cluster
2. 挑出明顯是迷因的（有固定句式 / 角色化標籤 / 高重複度）
3. 從該 cluster 找最有代表性的高讚留言當 `canonical_text`
4. 在劇本中找對應的幕，填 `scene_ref`
5. 寫到 `ground_truth.yaml`

**建議**：至少標 5–8 個 confirmed_memes，外加 1–2 個 potential。LLM 才有夠的範例可參考。

---

### 完整擴充流程

```bash
# 1. 把資料放好
cp your_comments.parquet  final_project/data/comments_with_embedding.parquet
cp your_ep1.txt           final_project/huan_scripts/ep1.txt
vim final_project/ground_truth.yaml          # 改成這集的迷因

# 2. 改 embed_scripts.py / embed_lines.py 加新集數
vim final_project/embed_scripts.py    # EPISODE_FILES 加新行
vim final_project/embed_lines.py      # 同上

# 3. 跑 embedding（需 GPU）
uv run final_project/embed_scripts.py --quantize
uv run final_project/embed_lines.py --quantize

# 4. 一鍵跑分析
uv run final_project/run_all.py

# 5. 開新 Claude Code session，貼 README「最後一步」的啟動 prompt
```

---

### 常見問題

**Q: 我沒有 ground truth，可以跑嗎？**

可以跑 `cluster.py` / `analyze.py` / `classify_quote_vs_remix.py` / `scene_heatmap.py`，這四個不需要 ground truth。但 `judge.py` 會缺 few-shot 範例，LLM 判斷品質會下降。建議至少手標 3 個再跑 judge。

**Q: 我的 embedding 不是 4096 維可以嗎？**

可以，但留言和劇本的 embedding 維度**必須一致**（同個模型）。如果你換成 1024 維的 model，留言和劇本都要重新 embed。

**Q: 我能用其他語言模型嗎？**

可以，但要注意：
- 留言和劇本必須用**同一個模型**編碼
- 中文語料效果 Qwen3-Embedding 較好，BGE-M3 也可
- 換完後 `architecture.md` 的「為什麼選 Qwen3」段落請對應更新
