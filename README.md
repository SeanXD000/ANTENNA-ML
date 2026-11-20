# antenna-ml

機器學習輔助 WiFi 6E/7 天線設計：用 CST 參數掃描結果訓練代理模型 (surrogate)，預測 S11 曲線並找出最佳設計參數。

## 檔案

| 檔案 | 用途 |
|---|---|
| `antenna_ml.py` | 主程式：讀 CST 匯出檔 → 集成模型訓練 → 留一驗證 → 細掃設計空間找最佳參數 → 主動學習推薦下一次 CST 模擬點 |
| `iterative_train.py` | 疊代微調：補跑 CST 後把新檔丟進 `data/`，在既有模型上微調，不用從頭訓練 |
| `compare_verify.py` | CST 驗證結果 vs 模型預測自動比對：印出覆蓋率、−10 dB 邊界、共振頻率、RMSE、不確定性校準，畫比對圖並追加紀錄到 `outputs/verify_log.csv` |
| `tolerance.py` | 製程容差分析：以 CST 實測過的設計為名目點做 Monte Carlo，估計尺寸誤差下的良率、單一參數敏感度，輸出 `outputs/tolerance_<檔名>.png` |
| `cst/` | `make_doe.py --batch N` 產生第 N 批多參數拉丁超立方取樣清單（`doe_batchN.csv`）與對應的 CST macro（第 1 批 `run_doe.bas`、之後 `run_doe_batchN.bas`），macro 自動逐組模擬並匯出到 `data/doe_batchN.txt`；`run_verify_tol03.bas` / `run_verify_tol05.bas` 跑 verify_best3 的 ±0.3 / ±0.5 mm 最差角落，輸出 `data/verify_tol03.txt` / `verify_tol05.txt` |
| `train.py` | 入門範例：最小的 PyTorch 代理模型（手打假資料），學習用 |
| `架構與目標.md` | 對照 ML 開發流程，說明每一步選了什麼方法、做到哪裡、下一步 |

## 環境

```bash
pip install torch numpy matplotlib
```

## 使用

1. 把 CST 參數掃描匯出的 `.txt` / `.dat` 放進 `data/`（格式：每組模擬以 `#Parameters = {...}` 開頭，接著是「頻率 GHz、S11 dB」兩欄）。
   - 目前的資料是槽孔天線（設計變數 `W_slot1`、`W_slot2`、`W_slot3`、`Ymax`，2–8 GHz）：4 個單參數掃描、`Ymax(12.5_18)` 補掃、4 參數 DOE 兩批各 25 組（`doe_batch1`、`doe_batch2`）、建議最佳點的 CST 驗證（`verify_best`、`verify_best2`、`verify_best3`），共 165 組；另有容差驗證 `verify_tol03`、`verify_tol05`（3 組，尚未納入訓練）。
   - **目前最佳設計**：`verify_best3` [W_slot1=36.97, W_slot2=12.502, W_slot3=22.944, Ymax=13.932]，CST 實測 6E 覆蓋率 100%、帶內最差點 −11.91 dB（邊界餘裕 1.91 dB）、−10 dB 頻寬 5.864–7.856 GHz。4 個參數各偏 ±0.3 mm 的最差角落 CST 實測 −11.32 dB，仍 100% 覆蓋；±0.5 mm 最差角落實測 98.5%（−9.37 dB），容差上限約 ±0.5 mm。
   - 舊天線的 `W4(0_13).txt` 參數名稱不同，預設會被略過（`--exclude`，預設 `W4*`）。若混入參數名稱不一致的檔案，程式會停下來並列出是哪些檔案。
2. 訓練 + 最佳化：

   ```bash
   python antenna_ml.py                          # 預設目標 6E 頻段
   python antenna_ml.py --band WIFI7             # 2G4 / 5G / 6E / WIFI7
   python antenna_ml.py --skip-validate          # 跳過留一驗證
   python antenna_ml.py --holdout W_slot2=12     # 指定驗證要藏的樣本（參數=數值）
   python antenna_ml.py --trust-radius 0.1       # 只搜尋離 CST 樣本 10% 參數範圍以內的組合
   python antenna_ml.py --search-scope all       # 舊搜尋範圍：所有 CST 樣本附近（預設 doe = 只在 DOE/verify 附近且不超出其範圍）
   ```

   產出：`antenna_surrogate.pth`（模型）、`outputs/validation.png`、`outputs/sweeps.png`（每條掃描線的 CST vs 模型熱圖）、`outputs/optimize_<band>.png`。
   未指定 `--epochs` 時會依資料量自動決定（每個模型約 1 萬次更新）；目前約 16 萬個頻點，對應 244 epochs。CPU 上整次約 45 分鐘，訓練時會印進度條與預估剩餘時間。
   留一驗證預設藏 18 條：單參數掃描檔各 1 條、每批 DOE 各 5 條、`verify_best*` 單點驗證檔整檔藏起來。
   信心區間（±σ）經過校正：σ = √[(s × 集成標準差)² + (β × 離最近訓練樣本的距離)²]，s、β 由留一驗證擬合並存在模型檔裡（目前 s = 2.55、β = 0）；用 `--skip-validate` 時不校正。
   4 個以上參數時，預設只在多參數資料（DOE、verify）涵蓋的範圍內搜尋：模型沿單參數掃描線外推到 DOE 範圍外時曾經嚴重高估。
3. 把建議點拿去 CST 模擬，匯出成 `data/verify_bestN.txt`（或跑新一批 DOE），**在重訓之前**先比對：

   ```bash
   python compare_verify.py data/verify_best3.txt --note "說明"
   python compare_verify.py data/verify_best3.txt --no-log       # 只看結果，不寫紀錄
   ```

   產出：`outputs/compare_<檔名>.png`、追加一列到 `outputs/verify_log.csv`。模型若比驗證檔新（可能已經用它訓練過），程式會警告。
4. 製程容差分析（選用）：

   ```bash
   python tolerance.py data/verify_best3.txt                     # 預設 ±0.05 / 0.1 / 0.2 mm
   python tolerance.py data/verify_best3.txt --tol 0.1 0.3 --n 5000
   ```

5. 比對完再重訓（`antenna_ml.py`）或微調：

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
- **2026-11-20** CST 驗證 ±0.5 mm 容差：新增 `cst/run_verify_tol05.bas`，跑 ±0.5 mm 估計最差的 2 個角落（兩種不同失效方式）。實測（`verify_tol05.txt`）：(−0.5, +0.5, −0.5, −0.5) = [36.47, 13.002, 22.444, 13.432] **6E 覆蓋率 98.5%、帶內最差 −9.37 dB**，−10 dB 下緣漂到 5.948 GHz（6E 下緣 5.925），容差估計 −9.97 dB，這種下緣漂移估計偏樂觀 0.6 dB；(+0.5, −0.5, −0.5, +0.5) = [37.47, 12.002, 22.444, 14.432] 100%、−10.28 dB（估 −10.16，偏保守，和 ±0.3 mm 同一種帶中凸起）。結論：±0.3 mm 確認安全，±0.5 mm 是容差上限，只有少數極端角落會掉覆蓋率（均勻 ±0.5 mm Monte Carlo 加 0.6 dB 懲罰後良率約 99.4%）。尚未重訓。
- **2026-11-20** `15cd08a` CST 確認容差分析：新增 `cst/run_verify_tol03.bas`，跑 verify_best3 的 ±0.3 mm 最差角落 [37.27, 12.202, 23.244, 14.232]（W_slot1 +0.3、W_slot2 −0.3、W_slot3 +0.3、Ymax +0.3）。實測（`verify_tol03.txt`）6E 覆蓋率 100%、帶內最差 −11.32 dB、邊界餘裕 1.32 dB，比 `tolerance.py` 估計的 −10.98 dB 好 0.34 dB；名目 → 角落實測只退 0.59 dB（模型估 0.93 dB），容差估計偏保守。`compare_verify.py`：帶內 RMSE 2.11 dB，主要是共振谷深度（實測 −37.6 vs 模型 −27.5 dB），−10 dB 邊界誤差 < 0.012 GHz。結論：verify_best3 在 ±0.3 mm 製程誤差內確認可用。尚未重訓。
- **2026-11-20** `8c88721` 新增 `tolerance.py` 製程容差分析：以 CST 實測設計為名目點，各設計變數加均勻分布 ±tol 誤差做 Monte Carlo；估計 S11 = CST 實測 + [模型(擾動) − 模型(名目)]，抵銷模型在名目點本身的偏差。verify_best3 在 ±0.05 / 0.1 / 0.2 / 0.3 mm 良率全部 100%，±0.3 mm 時帶內最差 S11 的 95 百分位 −11.39 dB（名目 −11.91）。最敏感的是 W_slot2（−1.40 dB/mm）與 Ymax（+0.66 dB/mm），W_slot1 幾乎無影響。±0.5 mm 的最差角落 −9.97 dB 才剛好過線。限制：不含板厚、介電常數；±0.3 mm 內是模型內插，未經 CST 確認。
- **2026-11-20** `7570ffc` 驗證新分數第 1 名並重訓，加進度條：[36.97, 12.502, 22.944, 13.932] 在 CST 實測（`verify_best3.txt`），重訓前用 `compare_verify.py` 比對：6E 覆蓋率 100%（預測 100%）、**邊界餘裕 1.91 dB**（預測 2.05，verify_best2 只有 0.46）、平均 −17.02 dB、−10 dB 頻寬 5.864–7.856 GHz（邊界誤差 < 0.08 GHz）；模型漏掉 6.04 GHz 的第二個共振谷，帶內 RMSE 2.53 dB，但誤差方向是保守的。**verify_best3 成為目前最佳設計。** 165 組重訓：18 個驗證點平均 AI 1.13 vs baseline 1.61 dB，s = 2.55。新前 5 名預測邊界餘裕 2.0–2.3 dB，只比實測多 0.1–0.4 dB（小於不確定性），設計已收斂。另外 `antenna_ml.py`、`iterative_train.py` 訓練時每 20% 印一行進度條（百分比、經過時間、預估剩餘），輸出導到檔案也逐行寫入。
- **2026-11-20** `0f61ba4` 檢討綜合分數：舊版 `覆蓋率×100 − 平均 dB` 讓 1 個百分點覆蓋率 = 1 dB 平均深度，深谷可以換掉覆蓋率（97.5% + 平均 −23 dB 贏過 100% + −17 dB）。新版 = `覆蓋率×70 + 最差子頻段覆蓋率×30 + 邊界餘裕 − 0.2 × 封頂平均 dB`：邊界餘裕是帶內最差點比 −10 dB 低多少（0–5 dB），只有 100% 覆蓋才會 > 0，所以永遠不會拿覆蓋率去換；平均 dB 每點最深只算到 −20 dB，只當 tie-breaker。中間試過只用封頂深度當次要項，結果前 5 名全部擠到 DOE 邊界 W_slot1=36.1，所以改用邊界餘裕。新前 5 名全部 6E 100%、邊界餘裕約 2 dB（verify_best2 實測 0.46 dB）、±0.5 dB，不貼邊界；第 1 名 [36.97, 12.502, 22.944, 13.932]。輸出加印邊界餘裕。
- **2026-11-19** `62c6dbd` 搜尋範圍限制在 DOE 涵蓋區域：新增 `--search-scope`（預設 `doe`），4 個以上參數時只以多參數資料（DOE、verify，共 52 組）為信任區域中心，並限制在它們的參數範圍內（W_slot1 36.1–44.8、W_slot2 10.2–18、W_slot3 18.1–25.8、Ymax 11.1–18.4）；`all` 為舊做法。用現有模型比較：舊範圍第 1、2 名落在 W_slot2 ≈ 7.5（不可信）；新範圍前 5 名全部預測 6E 100%、平均約 −16.4 dB、±0.36–0.92 dB，集中在 verify_best2 同一區。最穩的候選 [40.92, 13.685, 22.615, 13.925]（±0.36 dB）。模型未重訓，`outputs/optimize_6E.png` 要下次跑 `antenna_ml.py` 才會更新。
- **2026-11-19** `7ae0509` 第 2 批 DOE 與重訓：`cst/make_doe.py` 改為 `--batch N`，各批範圍與亂數種子寫在程式裡（第 1 批重產結果不變），並以 `run_doe.bas` 為範本產生 `run_doe_batch2.bas`。第 2 批 25 組範圍 W_slot1 36–42、W_slot2 10–16、W_slot3 20–26、Ymax 11–17。**重訓前**用 `compare_verify.py` 測舊模型：全頻 RMSE 平均 2.50 dB、15 組覆蓋率誤差超過 5 個百分點；Ymax < 13 的 8 組全部 2.8–5.9 dB、覆蓋率嚴重高估（例如實測 28% vs 預測 97.5%），舊建議點 [38.453, 13.049, 23.121, 12.843] 因此作廢。164 組重訓後：17 個驗證點平均 AI 1.09 vs baseline 1.59 dB，verify_best2 從 2.26 降到 0.97 dB，s = 2.25、β = 0。新第 1 名 [41.921, W_slot2=7.634, 23.075, 12.386] 在 DOE 範圍外、不可信；可信候選為 [39.948, 13.903, 22.391, 13.979]（預測 100%、±0.54 dB）。實測最好仍是 verify_best2（100%），第 2 批最好 #20 [40.6, 12.7, 22.1, 13.8]（97%）。下一步：搜尋限制在 DOE 涵蓋範圍、檢討綜合分數。
- **2026-11-19** `4fdcbbb` 不確定性加入距離項並擴大驗證集：σ = √[(s × 集成標準差)² + (β × d)²]，d = 離最近訓練樣本的正規化距離，s、β 以驗證樣本的高斯負對數概似擬合；模型檔另存訓練樣本位置，`iterative_train.py` 微調後同步更新。留一驗證從 6 條增為 12 條（DOE 藏 5 條、`verify_best*` 整檔藏起來）。結果：**s = 1.75、β = 0**，誤差和距離無關（verify_best2 距離只有 0.068 但誤差最大 2.26 dB），距離項目前沒作用；校正後 verify_best2 從低估 7.5 倍降到 1.5 倍。多藏 DOE 點後發現 DOE 區域真實誤差 0.5–2 dB（平均 1.35 vs baseline 2.24），先前的 0.34 dB 只是單點剛好準；verify_best2 是唯一輸給 baseline 的點。建議點不變。下一步改為補第二批 DOE（W_slot1 36–42）。
- **2026-11-19** `30dd0e5` 不確定性校正：集成標準差乘上校正係數 `sigma_scale`，在留一驗證時用「驗證樣本實際誤差均方根 ÷ 集成標準差均方根」估計（限 1–10 倍），存進模型檔，所有預測自動套用（舊模型檔讀入時視為 1.0）。重訓後係數 **1.84**，預測與建議點不變，最佳點不確定性 ±0.69 → ±1.26 dB。限制：只乘一個整體倍數，外推區域仍低估（verify_best2 實際差 7.5 倍）；下一步改為依離 CST 樣本的距離放大不確定性。README 補上 `compare_verify.py` 與驗證流程。
- **2026-11-19** `17bd3c1` 新增 `compare_verify.py`：讀 CST 驗證檔，自動比對模型預測的覆蓋率、平均 S11、−10 dB 邊界、共振頻率、全頻／帶內 RMSE 與不確定性校準，輸出比對圖並追加到 `outputs/verify_log.csv`。模型比驗證檔新時會警告（實測 verify_best2 重訓後 RMSE 只剩 0.13 dB，原本是 2.26 dB，所以必須在重訓前比對）。
- **2026-11-18** `6ca96f9` 驗證第一個建議最佳點並重新訓練：把 [W_slot1=38.92, W_slot2=12.9, W_slot3=23.196, Ymax=14.725] 在 CST 實測（`verify_best2.txt`，資料不進 git）。6E 覆蓋率 **100%**（與預測一致），平均 −16.01 dB（預測 −16.58），−10 dB 區段 5.786–7.724 GHz；深谷預測過頭（−56 vs −38 dB），全頻 RMSE 2.26 dB 但模型自報只有 ±0.30 dB，信心區間偏窄。納入訓練後共 139 組重訓：驗證平均 RMSE AI 0.224 dB vs 內插 baseline 0.518 dB，DOE 點 0.34 dB。新建議點 [38.453, 13.049, 23.121, 12.843] 預測平均 −18.68 dB、頻寬 5.702–7.766 GHz，尚未驗證。`架構與目標.md` 同步更新，下一步以校正不確定性為主。
- **2026-11-18** `8fb186e` 驗證時加入內插 baseline 比較：驗證點在掃描線上用相鄰兩條曲線線性內插，沒有掃描線可用（DOE 點）則取最近 4 條曲線加權平均；每個驗證點印出 AI 與基準的 RMSE，並印平均。六組驗證點 AI 都優於基準，DOE 點 0.32 dB 對 0.68 dB；Ymax 兩點差距很小。同步更新 `架構與目標.md`：資料增為 138 條、baseline 標為完成，下一步改為驗證建議最佳點、定義門檻、多藏 DOE 驗證點。
- **2026-11-18** `5442b25` 新增 4 參數聯合取樣：`cst/make_doe.py` 以拉丁超立方在目前最佳點附近產生 25 組（`cst/doe_batch1.csv`），`cst/run_doe.bas` 是 CST macro，自動逐組模擬並匯出到 `data/doe_batch1.txt`。加入訓練後共 138 組；4 個參數同時變動的驗證點 RMSE 0.32 dB、6E 覆蓋率預測與真實一致（76.5%）。新建議最佳點 [W_slot1=38.92, W_slot2=12.9, W_slot3=23.196, Ymax=14.725] 預測 6E 覆蓋率 100%、不確定性 ±0.30 dB，但 W_slot1 落在取樣範圍（40–45）之外，需回 CST 驗證。
- **2026-11-11** `9f7d34e` 改用槽孔天線的 4 個單參數掃描（`Wslot1`、`Wslot2`、`Wslot3`、`Ymax`，2–8 GHz）訓練。舊天線的 `W4(0_13).txt` 預設排除（`--exclude`）；混入參數名稱不一致的檔案時，程式會列出檔名並停止；交會點重複的模擬只保留一份。其他調整：留一驗證改為每個掃描檔各藏中間 1 組，`--holdout` 改用「參數=數值」；epochs 依資料量自動決定（目前 400）；4 個以上參數時只在 CST 樣本附近搜尋（`--trust-radius`）；新增 `outputs/sweeps.png`。驗證 RMSE 為 0.05–0.51 dB。推薦的最佳點同時改了 4 個參數、不確定性 ±3.7 dB，需要回 CST 驗證。
- **2026-11-10** `91f528c` 練習上傳流程：把前 3 筆變更紀錄寫進 README。
- **2026-11-10** `941fa0c` 新增 `架構與目標.md`：把通用 ML 開發流程（定義問題 → 部署監控）逐步對照到本專案，標出每一步的做法與完成狀態，並補上本專案特有的「第 8 步：用模型做設計最佳化」。文件也記錄目前的驗證結果（RMSE 約 0.6 dB）、過擬合跡象與信心區間偏窄的問題，以及依優先順序排列的下一步。README 檔案表加入連結。
- **2026-11-10** `bf9a385` README 新增「變更紀錄」區塊，約定每次 commit 後由 Claude 讀取變更內容整理。原本規劃的自動 hook 方案（只能記錄行數）沒有採用。
- **2026-11-10** `4952fb5` 初始 commit：整合原資料夾的 V1/V2 主程式與兩版疊代訓練，詳見上方「整合紀錄」。
<!-- CHANGELOG:END -->
