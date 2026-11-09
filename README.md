# antenna-ml

機器學習輔助 WiFi 6E/7 天線設計：用 CST 參數掃描結果訓練代理模型 (surrogate)，預測 S11 曲線並找出最佳設計參數。

## 檔案

| 檔案 | 用途 |
|---|---|
| `antenna_ml.py` | 主程式：讀 CST 匯出檔 → 集成模型訓練 → 留一驗證 → 細掃設計空間找最佳參數 → 主動學習推薦下一次 CST 模擬點 |
| `iterative_train.py` | 疊代微調：補跑 CST 後把新檔丟進 `data/`，在既有模型上微調，不用從頭訓練 |
| `train.py` | 入門範例：最小的 PyTorch 代理模型（手打假資料），學習用 |
| `架構與目標.md` | 對照 ML 開發流程，說明每一步選了什麼方法、做到哪裡、下一步 |

## 環境

```bash
pip install torch numpy matplotlib
```

## 使用

1. 把 CST 參數掃描匯出的 `.txt` / `.dat` 放進 `data/`（格式：每組模擬以 `#Parameters = {...}` 開頭，接著是「頻率 GHz、S11 dB」兩欄）。
2. 訓練 + 最佳化：

   ```bash
   python antenna_ml.py                  # 預設目標 6E 頻段
   python antenna_ml.py --band WIFI7     # 2G4 / 5G / 6E / WIFI7
   python antenna_ml.py --skip-validate  # 跳過留一驗證
   ```

   產出：`antenna_surrogate.pth`（模型）、`outputs/validation.png`、`outputs/optimize_<band>.png`。
3. 補跑 CST 後微調：

   ```bash
   python iterative_train.py             # 預設 300 epochs、lr 5e-5
   ```

`data/`、`outputs/`、`*.pth` 都在 `.gitignore` 內，不會上傳 GitHub；換電腦需自行複製資料，或重跑 `antenna_ml.py` 產生模型。

## 整合紀錄（2026-11-10）

原始資料夾完整備份於 `C:\attena\antenna-ml_backup_20261110\`。

| 原檔案 | 處理 |
|---|---|
| `antenna_ml_r2bygemini.py` (V2) | **改名為 `antenna_ml.py`**。它包含 V1 全部功能，另加多參數、Ensemble 信心區間、主動學習、物理感知 loss、多頻段平衡評分、GPU 支援。修正：docstring 用法寫成不存在的 `antenna_ml_v2.py`、loss 註解寫「Smooth L1 混合」但實際只是加權 MSE；模型檔與輸出圖去掉 `_v2` 後綴 |
| `antenna_ml.py` (V1) | 移除（被 V2 取代） |
| `iterative_train.py` + `iterative_train_R1.py` | **合併重寫為新的 `iterative_train.py`**。舊版只吃手打 2 筆資料；R1 讀 `data/` 但假設欄位叫 `length/width`，與 CST 實際格式不符、無法執行。新版直接 import `antenna_ml.py` 的解析器與模型，微調同一個 `antenna_surrogate.pth`。實測預設 lr 用 5e-5（2e-4 會讓已收斂模型 RMSE 從約 0.02 dB 升到約 0.04 dB） |
| `train.py` | 保留為入門範例，刪掉除錯用 `print("123")` |
| `antenna_surrogate_v2.pth` | 改名為 `antenna_surrogate.pth`（與 V2 程式相容，可直接給 `iterative_train.py` 用） |
| `antenna_surrogate_w4.pth`、`antenna_surrogate_model.pth` | 移除（分別屬於 V1 與舊玩具模型，新程式不再讀取） |
| `outputs/*_v2.png` | 改名去掉 `_v2`；V1 舊圖移除 |
| `test.py`、`pytest.ai`、`R2新增功能筆記.txt` | 移除（只有 `print("123")` 或空檔） |
| `__pycache__/` | 移除；`.vscode/` 保留在本機但不進 git |

## 變更紀錄

每次 commit 後由 Claude 讀取變更內容整理，新的在最上面。

<!-- CHANGELOG:START -->
<!-- CHANGELOG:END -->
