"""
機器學習輔助 WiFi 6E/7 天線設計最佳化
------------------------------------------------
[整合說明] 本檔 = 舊版 antenna_ml_r2bygemini.py (V2)，它完整涵蓋舊 antenna_ml.py (V1) 的功能，
          V1 已移除（備份在 ../antenna-ml_backup_20261110/）。詳見 README.md「整合紀錄」。

流程：
 1. 讀取 CST 參數掃描匯出的 .txt（一個檔案含多組 #Parameters 區塊）
    預設排除舊天線的 W4*.txt（參數名稱不同，不能混在一起訓練），見 EXCLUDE_PATTERNS / --exclude
 2. 自動找出有變動的設計參數（目前為 W_slot1, W_slot2, W_slot3, Ymax）當作模型輸入
 3. 訓練代理模型 (Surrogate)： [設計參數, 頻率] -> S11 (dB)
 4. 留一驗證：藏起部分資料不訓練，看模型能否預測出那條曲線
 5. 用代理模型在連續的參數空間細掃，找出 WiFi 頻段表現最好的設計

V2 相對 V1 的升級重點：
 1. 支援「多設計參數（多個溝槽數值）」同時訓練、網格/拉丁超立方搜尋與最佳化
 2. 模型集成 (Ensemble)：同時訓練 N 個模型，輸出預測平均值與「信心區間（標準差陰影帶）」
 3. 主動學習 (Active Learning) 推薦：自動找出「高潛力且模型最不確定」的參數組，指導下一次 CST 模擬
 4. 天線物理感知 Loss：強化 -10 dB 頻寬邊界與目標頻段權重，衰減 <-25 dB 極端深谷的過度懲罰
 5. 多頻段平衡評分：修正 WIFI7 三頻段因頻點數不同導致偏心 6E 而犧牲 2.4G 的問題
 6. 支援 GPU (CUDA) 自動加速與 Normalizer 防除零保護

資料形狀注意：目前 data/ 是 4 個「一次只改一個參數」的掃描（每個檔案掃一個參數，其餘固定），
在 4 維參數空間裡只是幾條線，不是填滿的網格。所以 4 個參數以上時，最佳化只在「已模擬過的點附近」
搜尋（--trust-radius），離資料太遠的組合模型沒有依據，預測不可信。

用法：
  python antenna_ml.py                         # 驗證 + 訓練 + 最佳化 (自動支援單/多參數)
  python antenna_ml.py --band WIFI7            # 目標頻段: 6E / 5G / 2G4 / WIFI7
  python antenna_ml.py --holdout W_slot2=12 Ymax=6   # 指定驗證時要藏起來的樣本（參數=數值）
  python antenna_ml.py --n-ensemble 5          # 使用 5 個模型集成並繪製不確定性陰影帶
  python antenna_ml.py --skip-validate         # 直接用全部資料訓練
  python antenna_ml.py --exclude "W4*" "old_*" # 排除的檔名樣式（預設 W4*）；只寫 --exclude 不帶值 = 全部讀
  python antenna_ml.py --epochs 300            # 指定 epochs；不指定則依資料量自動決定
"""
import argparse
import fnmatch
import glob
import itertools
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

DATA_DIR = "./data"
OUT_DIR = "./outputs"
MODEL_PATH = "antenna_surrogate.pth"  # iterative_train.py 也讀寫這個檔
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 預設不讀的檔案：W4*.txt 是舊天線（參數 H, L, W, W4...），和現在的槽孔天線（W_slot1/2/3, Ymax...）不是同一個設計
EXCLUDE_PATTERNS = ["W4*"]
TARGET_STEPS = 10000  # 未指定 --epochs 時，每個模型的目標更新步數（舊 W4 資料 2500 epochs ≈ 1 萬步）

# WiFi 頻段 (GHz)
WIFI_BANDS = {
    "2G4": [(2.400, 2.4835)],
    "5G": [(5.150, 5.850)],
    "6E": [(5.925, 7.125)],
    "WIFI7": [(2.400, 2.4835), (5.150, 5.850), (5.925, 7.125)],
}
S11_GOAL_DB = -10.0  # 一般天線匹配標準：S11 < -10 dB

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. 讀取 CST 文字檔
# ============================================================
def parse_cst_sweep(file_path):
    """解析 CST 參數掃描匯出檔，回傳 list[{"params": dict, "freq": array, "s11": array}]"""
    runs, current = [], None
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#Parameters"):
                m = re.search(r"\{(.*)\}", line)
                if not m:
                    continue
                body = m.group(1)
                params = {}
                for item in body.split(";"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        try:
                            params[k.strip()] = float(v)
                        except ValueError:
                            pass
                current = {"params": params, "freq": [], "s11": []}
                runs.append(current)
            elif line.startswith("#"):
                continue  # 欄位名稱、分隔線
            elif current is not None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        current["freq"].append(float(parts[0]))
                        current["s11"].append(float(parts[1]))
                    except ValueError:
                        pass
    valid_runs = []
    for r in runs:
        if len(r["freq"]) > 0:
            r["freq"] = np.array(r["freq"])
            r["s11"] = np.array(r["s11"])
            valid_runs.append(r)
    return valid_runs


def load_all_runs(data_dir, exclude=None):
    """讀取資料夾內所有 CST 匯出檔。exclude：要略過的檔名樣式（fnmatch），None = 用 EXCLUDE_PATTERNS。
    每組模擬會加上 "source"（來源檔名，不含副檔名），用來分組驗證與畫圖。"""
    exclude = EXCLUDE_PATTERNS if exclude is None else exclude
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")) + glob.glob(os.path.join(data_dir, "*.dat")))
    skipped = [fp for fp in files if any(fnmatch.fnmatch(os.path.basename(fp), pat) for pat in exclude)]
    files = [fp for fp in files if fp not in skipped]
    for fp in skipped:
        print(f"略過 {os.path.basename(fp)}（符合排除樣式 {exclude}）")
    if not files:
        raise FileNotFoundError(f"在 '{data_dir}' 找不到任何 .txt / .dat 檔案（排除樣式 {exclude}）")
    runs = []
    for fp in files:
        rs = parse_cst_sweep(fp)
        src = os.path.splitext(os.path.basename(fp))[0]
        for r in rs:
            r["source"] = src
        print(f"讀取 {os.path.basename(fp)}：{len(rs)} 組模擬")
        runs.extend(rs)
    if not runs:
        raise ValueError("讀取到的模擬資料為空，請確認 CST 匯出檔格式是否包含 #Parameters {...}")

    # 所有檔案必須是同一個天線（參數名稱相同），否則無法放進同一個模型
    key_sets = {}
    for r in runs:
        key_sets.setdefault(tuple(sorted(r["params"])), set()).add(r["source"])
    if len(key_sets) > 1:
        detail = "\n".join(f"  {sorted(srcs)}: {list(keys)}" for keys, srcs in key_sets.items())
        raise ValueError("不同檔案的參數名稱不一致，看起來是不同的天線設計，不能一起訓練：\n"
                         f"{detail}\n請用 --exclude 排除其中一組（例如 --exclude \"W4*\"）。")

    # 不同掃描檔可能在交會點重複模擬同一組參數（例如 W_slot2=14 & W_slot3=8），只保留一份
    seen, unique = {}, []
    for r in runs:
        key = tuple(sorted(r["params"].items()))
        if key in seen:
            print(f"  {r['source']} 有一組參數與 {seen[key]} 重複，只保留一份")
            continue
        seen[key] = r["source"]
        unique.append(r)
    runs = unique

    # 自動找出所有出現過的參數，並判斷有變動的參數 => 設計變數（如多個溝槽數值）
    all_keys = sorted({k for r in runs for k in r["params"]})
    design_params = [k for k in all_keys if len({r["params"].get(k) for r in runs}) > 1]
    fixed = {k: runs[0]["params"].get(k) for k in all_keys if k not in design_params}
    return runs, design_params, fixed


# ============================================================
# 2. 代理模型與物理感知訓練 (Ensemble + Task-Aware Loss)
# ============================================================
class AntennaSurrogate(nn.Module):
    """輸入：正規化後的設計參數(支援多維溝槽參數) + 頻率；輸出：S11 (dB)
    頻率與參數皆用 Fourier features 編碼，讓網路能畫出尖銳的共振谷。"""

    def __init__(self, n_params, n_freq_feat=48, n_param_feat=3, hidden=192, depth=4):
        super().__init__()
        self.register_buffer("kf", torch.arange(1, n_freq_feat + 1, dtype=torch.float32) * np.pi)
        self.register_buffer("kp", torch.arange(1, n_param_feat + 1, dtype=torch.float32) * np.pi)
        in_dim = 1 + 2 * n_freq_feat + n_params * (1 + 2 * n_param_feat)
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def encode(self, p, f):
        ff = f * self.kf  # (N, K)
        pp = p.unsqueeze(-1) * self.kp  # (N, P, K)
        return torch.cat(
            [f, torch.sin(ff), torch.cos(ff), p,
             torch.sin(pp).flatten(1), torch.cos(pp).flatten(1)], dim=1)

    def forward(self, p, f):
        return self.net(self.encode(p, f))


class Normalizer:
    """把參數與頻率縮到 [0,1]，S11 標準化（含 +1e-8 防除零保護）。"""

    def __init__(self, p_min, p_max, f_min, f_max, y_mean, y_std):
        self.p_min = np.asarray(p_min, float)
        self.p_max = np.asarray(p_max, float)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std) if float(y_std) > 1e-8 else 1.0

    def p(self, x):
        span = np.where((self.p_max - self.p_min) > 1e-8, self.p_max - self.p_min, 1.0)
        return (np.asarray(x, float) - self.p_min) / span

    def f(self, x):
        span = (self.f_max - self.f_min) if (self.f_max - self.f_min) > 1e-8 else 1.0
        return (np.asarray(x, float) - self.f_min) / span

    def state(self):
        return dict(p_min=self.p_min.tolist(), p_max=self.p_max.tolist(), f_min=self.f_min,
                    f_max=self.f_max, y_mean=self.y_mean, y_std=self.y_std)


def runs_to_arrays(runs, design_params):
    P, F, Y = [], [], []
    for r in runs:
        pv = [r["params"][k] for k in design_params]
        P.append(np.tile(pv, (len(r["freq"]), 1)))
        F.append(r["freq"])
        Y.append(r["s11"])
    return np.concatenate(P), np.concatenate(F), np.concatenate(Y)


def compute_sample_weights(F, Y, bands):
    """電磁物理感知權重：
    1. 目標 WiFi 頻段內的頻點給予 2.0 倍權重
    2. 接近 -10 dB 頻寬邊界 (-14 dB ~ -6 dB) 給予 1.5 倍權重
    3. 極端深谷 (< -25 dB) 權重衰減為 0.4 倍，避免模型浪費容量硬追無意義的深谷
    """
    w = np.ones_like(Y, dtype=np.float32)
    in_band = np.zeros_like(F, dtype=bool)
    for lo, hi in bands:
        in_band |= (F >= lo) & (F <= hi)
    w[in_band] *= 2.0
    near_threshold = (Y >= -14.0) & (Y <= -6.0)
    w[near_threshold] *= 1.5
    deep_null = Y < -25.0
    w[deep_null] *= 0.4
    return w / np.mean(w)


def auto_epochs(n_points, bs=4096, target_steps=TARGET_STEPS):
    """依資料量決定 epochs：讓每個模型大約更新 target_steps 次。
    資料多時每個 epoch 步數多，epochs 就少；避免 10 萬頻點 × 2500 epochs 在 CPU 上跑好幾小時。"""
    steps_per_epoch = max(1, int(np.ceil(n_points / min(bs, n_points))))
    return int(min(2500, max(200, round(target_steps / steps_per_epoch))))


def train_ensemble(runs, design_params, bands, epochs=None, n_ensemble=5, verbose=True):
    """訓練多個代理模型組成 Ensemble，用於預測平均值與不確定性（標準差）。
    epochs=None 時依資料量自動決定（見 auto_epochs）。"""
    P, F, Y = runs_to_arrays(runs, design_params)
    if epochs is None:
        epochs = auto_epochs(len(Y))
    norm = Normalizer(P.min(0), P.max(0), F.min(), F.max(), Y.mean(), Y.std())
    weights = compute_sample_weights(F, Y, bands)

    p_t = torch.tensor(norm.p(P), dtype=torch.float32, device=DEVICE)
    f_t = torch.tensor(norm.f(F), dtype=torch.float32, device=DEVICE).unsqueeze(1)
    y_t = torch.tensor((Y - norm.y_mean) / norm.y_std, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    w_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    models = []
    n, bs = len(y_t), min(4096, len(y_t))

    for m_idx in range(n_ensemble):
        seed = 42 + m_idx * 100
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = AntennaSurrogate(len(design_params)).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

        for ep in range(epochs):
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                pred = model(p_t[idx], f_t[idx])
                # 加權 MSE 損失（權重見 compute_sample_weights）
                diff = (pred - y_t[idx]) ** 2
                loss = torch.mean(w_t[idx] * diff)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            full_mse = nn.functional.mse_loss(model(p_t, f_t), y_t).item()
            rmse_db = np.sqrt(full_mse) * norm.y_std
        if verbose:
            print(f"  [模型 {m_idx + 1}/{n_ensemble}] 訓練完成 ({epochs} epochs) | 全頻訓練 RMSE = {rmse_db:.3f} dB")
        models.append(model)

    return models, norm


@torch.no_grad()
def predict_curves_batch(models, norm, param_grid, freq):
    """批次預測多組參數的 S11 曲線。
    param_grid: shape (M, n_params)
    回傳:
      mean_curves: shape (M, len(freq))
      std_curves:  shape (M, len(freq))  <- 模型不確定性 (dB)
    """
    param_grid = np.atleast_2d(np.asarray(param_grid, dtype=float))
    M = len(param_grid)
    K = len(freq)

    # 展開成 (M * K, n_params) 與 (M * K, 1)
    P_rep = np.repeat(norm.p(param_grid), K, axis=0)
    F_tile = np.tile(norm.f(freq), M)

    p_t = torch.tensor(P_rep, dtype=torch.float32, device=DEVICE)
    f_t = torch.tensor(F_tile, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    preds_all = []
    batch_limit = 65536
    for model in models:
        out_list = []
        for i in range(0, len(p_t), batch_limit):
            out = model(p_t[i:i + batch_limit], f_t[i:i + batch_limit]).squeeze(1).cpu().numpy()
            out_list.append(out)
        preds = np.concatenate(out_list, axis=0) * norm.y_std + norm.y_mean
        preds_all.append(preds.reshape(M, K))

    preds_stack = np.stack(preds_all, axis=0)  # (n_ensemble, M, K)
    return np.mean(preds_stack, axis=0), np.std(preds_stack, axis=0)


def predict_single(models, norm, param_values, freq):
    mean_c, std_c = predict_curves_batch(models, norm, [param_values], freq)
    return mean_c[0], std_c[0]


# ============================================================
# 3. 多頻段平衡評分與主動學習指標
# ============================================================
def band_metrics(freq, s11, bands):
    """針對每個子頻段獨立計算覆蓋率與平均 S11，避免寬頻段(6E)稀釋窄頻段(2.4G)。"""
    sub_covs, sub_means, sub_worsts, sub_bests = [], [], [], []
    for lo, hi in bands:
        mask = (freq >= lo) & (freq <= hi)
        s = s11[mask]
        if len(s) == 0:
            continue
        sub_covs.append(float(np.mean(s < S11_GOAL_DB)))
        sub_means.append(float(np.mean(s)))
        sub_worsts.append(float(np.max(s)))
        sub_bests.append(float(np.min(s)))

    return dict(
        coverage=float(np.mean(sub_covs)),      # 各子頻段覆蓋率的平均（權重相等）
        min_coverage=float(np.min(sub_covs)),   # 最差子頻段的覆蓋率（木桶效應指標）
        sub_coverages=sub_covs,
        mean_db=float(np.mean(sub_means)),
        worst_db=float(np.max(sub_worsts)),
        best_db=float(np.min(sub_bests)),
    )


def score(m):
    """綜合評分：優先要求各子頻段均衡覆蓋，其次看平均覆蓋率與深度。"""
    return m["coverage"] * 70.0 + m["min_coverage"] * 30.0 - m["mean_db"]


def resonances(freq, s11, thr=S11_GOAL_DB):
    """找出 S11 < thr 的區段，回傳 [(f_low, f_high, f_min, s11_min)]"""
    out, below = [], s11 < thr
    i = 0
    while i < len(freq):
        if below[i]:
            j = i
            while j + 1 < len(freq) and below[j + 1]:
                j += 1
            k = i + np.argmin(s11[i:j + 1])
            out.append((freq[i], freq[j], freq[k], s11[k]))
            i = j + 1
        else:
            i += 1
    return out


def format_params(design_params, values):
    return ", ".join(f"{k}={v:g}" for k, v in zip(design_params, values))


def generate_search_grid(runs, design_params, max_total_points=5000, trust_radius=0.15):
    """自動根據參數數量建立搜尋網格（支援 1D 細掃、2D/3D 網格、高維度局部取樣）。
    4 個參數以上：只在「已模擬過的點附近」取樣（正規化到 [0,1] 後，距離 ≤ trust_radius），
    因為一次掃一個參數的資料只覆蓋幾條線，離線太遠的組合模型是在瞎猜。"""
    n_p = len(design_params)
    bounds = []
    for k in design_params:
        vals = [r["params"][k] for r in runs]
        bounds.append((min(vals), max(vals)))

    if n_p == 1:
        lo, hi = bounds[0]
        grid_1d = np.round(np.arange(lo, hi + 1e-9, 0.05), 3)
        return grid_1d.reshape(-1, 1), [grid_1d]
    elif n_p <= 3:
        steps_per_dim = int( round(max_total_points ** (1.0 / n_p)) )
        steps_per_dim = max(15, min(60, steps_per_dim))
        axes = [np.round(np.linspace(lo, hi, steps_per_dim), 3) for lo, hi in bounds]
        combos = np.array(list(itertools.product(*axes)), dtype=float)
        return combos, axes
    else:
        # 4 個參數以上：以每個 CST 樣本為中心做局部隨機擾動（信任區域），並包含樣本點本身
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = np.where(hi - lo > 1e-8, hi - lo, 1.0)
        centers = np.array([[r["params"][k] for k in design_params] for r in runs], dtype=float)
        n_rand = max(0, max_total_points - len(centers))
        picks = centers[rng.integers(0, len(centers), size=n_rand)]
        # 在正規化空間的球內均勻取樣（半徑 trust_radius）：同時改多個參數時，總偏移量仍受限
        direc = rng.normal(size=(n_rand, n_p))
        direc /= np.linalg.norm(direc, axis=1, keepdims=True)
        radius = trust_radius * rng.uniform(size=(n_rand, 1)) ** (1.0 / n_p)
        offsets = direc * radius * span
        combos = np.vstack([centers, np.clip(picks + offsets, lo, hi)])
        return np.round(combos, 3), None


def parse_holdout(specs, design_params):
    """把 ["W_slot2=12", "Ymax=6"] 轉成 [("W_slot2", 12.0), ("Ymax", 6.0)]。
    只有一個設計變數時，也接受純數字（舊用法 --holdout 6 10）。"""
    out = []
    for s in specs:
        if "=" in s:
            k, v = s.split("=", 1)
            k = k.strip()
            if k not in design_params:
                raise SystemExit(f"--holdout 的參數 {k} 不是設計變數 {design_params}")
            out.append((k, float(v)))
        elif len(design_params) == 1:
            out.append((design_params[0], float(s)))
        else:
            raise SystemExit(f"多參數時 --holdout 要寫成 參數=數值，例如 {design_params[0]}=10")
    return out


def group_by_source(runs):
    groups = {}
    for r in runs:
        groups.setdefault(r.get("source", ""), []).append(r)
    return groups


def swept_param(rs, design_params):
    """一個掃描檔裡真正有在變的參數（一次只掃一個參數時就是那一個）。"""
    varying = [k for k in design_params if len({r["params"][k] for r in rs}) > 1]
    return varying[0] if varying else design_params[0]


def default_holdout(runs, design_params):
    """自動挑驗證樣本：每個來源檔（每條掃描線）各藏 1 組位於中間的樣本，避免選到邊界外插。
    只有一個來源檔時，沿用舊做法：排序後 1/3 與 2/3 位置各藏 1 組。"""
    groups = group_by_source(runs)
    key = lambda r: [r["params"][k] for k in design_params]
    if len(groups) == 1:
        rs = sorted(runs, key=key)
        return [rs[i] for i in sorted({len(rs) // 3, (len(rs) * 2) // 3})]
    held = []
    for src in sorted(groups):
        rs = groups[src]
        k = swept_param(rs, design_params)
        rs = sorted(rs, key=lambda r: r["params"][k])
        if len(rs) >= 3:
            held.append(rs[len(rs) // 2])
    return held


def plot_sweeps(runs, design_params, models, norm, bands, out_path):
    """每個來源檔（一條掃描線）畫一列：左 = CST 原始 S11 熱圖，右 = 代理模型在同一條線上的預測。
    兼做資料探索 (EDA)：看共振頻率怎麼隨參數移動。"""
    groups = group_by_source(runs)
    n = len(groups)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.4 * n), squeeze=False)
    for row, src in enumerate(sorted(groups)):
        rs = groups[src]
        k = swept_param(rs, design_params)
        rs = sorted(rs, key=lambda r: r["params"][k])
        vals = np.array([r["params"][k] for r in rs])
        freq = rs[0]["freq"]
        raw = np.array([np.interp(freq, r["freq"], r["s11"]) for r in rs])
        fine_vals = np.linspace(vals[0], vals[-1], 120)
        base = np.array([rs[len(rs) // 2]["params"][d] for d in design_params], dtype=float)
        grid = np.tile(base, (len(fine_vals), 1))
        grid[:, design_params.index(k)] = fine_vals
        pred, _ = predict_curves_batch(models, norm, grid, freq)
        fixed_txt = ", ".join(f"{d}={base[i]:g}" for i, d in enumerate(design_params) if d != k)
        for col, (data, yv, title) in enumerate([(raw, vals, "CST 原始資料"), (pred, fine_vals, "代理模型預測")]):
            ax = axes[row, col]
            im = ax.pcolormesh(freq, yv, data, cmap="viridis_r", vmin=-25, vmax=0, shading="nearest")
            for lo, hi in bands:
                ax.axvline(lo, color="w", ls="--", lw=0.8)
                ax.axvline(hi, color="w", ls="--", lw=0.8)
            ax.set_ylabel(k)
            ax.set_title(f"{src}：{title}（{fixed_txt}）", fontsize=9)
            if row == n - 1:
                ax.set_xlabel("Frequency (GHz)")
            fig.colorbar(im, ax=ax, label="S11 (dB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================
def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR, help="CST 匯出檔資料夾路徑")
    ap.add_argument("--band", default="6E", choices=list(WIFI_BANDS), help="目標最佳化頻段")
    ap.add_argument("--holdout", nargs="*", default=None,
                    help="驗證時要隱藏的樣本，寫成 參數=數值（如 W_slot2=12）；未指定則每個掃描檔各藏中間 1 組")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help=f"略過的檔名樣式（預設 {EXCLUDE_PATTERNS}）；只寫 --exclude 不帶值 = 全部讀取")
    ap.add_argument("--epochs", type=int, default=None, help="訓練 Epochs 數（預設依資料量自動決定）")
    ap.add_argument("--trust-radius", type=float, default=0.15,
                    help="4 個以上參數時，只搜尋離 CST 樣本正規化距離 ≤ 此值的組合（0.15 ≈ 參數範圍的 15%%）")
    ap.add_argument("--n-ensemble", type=int, default=5, help="集成模型數量 (計算信心區間)")
    ap.add_argument("--skip-validate", action="store_true", help="跳過留一驗證")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    bands = WIFI_BANDS[args.band]

    print(f"使用運算裝置：{DEVICE}")
    runs, design_params, fixed = load_all_runs(args.data, args.exclude)
    freq = runs[0]["freq"]
    print(f"共 {len(runs)} 組模擬，每組 {len(freq)} 個頻點 ({freq[0]:.2f}-{freq[-1]:.2f} GHz)")
    if args.epochs is None:
        n_points = sum(len(r["freq"]) for r in runs)
        print(f"未指定 --epochs，依資料量 ({n_points} 個頻點) 自動設為約 {auto_epochs(n_points)} epochs")
    print(f"自動偵測到的設計變數 (共 {len(design_params)} 個)：{design_params}")
    print(f"固定不變的參數：{fixed}")

    # 印出各設計變數的取樣範圍
    for k in design_params:
        u_vals = sorted({r["params"][k] for r in runs})
        print(f"  參數 [{k}] 取樣值 ({len(u_vals)} 種): {u_vals}")
    print()

    # ---- 原始模擬資料在目標頻段的表現 ----
    print(f"=== 原始 CST 資料在 {args.band} 頻段 {bands} 表現最好的 10 組 ===")
    sorted_runs = sorted(runs, key=lambda r: -score(band_metrics(r["freq"], r["s11"], bands)))
    for r in sorted_runs[:10]:
        m = band_metrics(r["freq"], r["s11"], bands)
        p_str = format_params(design_params, [r["params"][k] for k in design_params])
        print(f"  [{p_str}]  均衡覆蓋率={m['coverage'] * 100:5.1f}% (最低子頻段={m['min_coverage'] * 100:4.1f}%)  "
              f"平均={m['mean_db']:6.2f} dB  最低={m['best_db']:6.2f} dB")
    if len(sorted_runs) > 10:
        print(f"  ... (其餘 {len(sorted_runs) - 10} 組省略)")
    print()

    # ---- 留一驗證 (支援單參數與多參數 + 繪製信心區間陰影帶) ----
    if not args.skip_validate and len(runs) >= 4:
        if args.holdout:
            # 樣本只要符合任一個 參數=數值 就藏起來。注意多參數時像 W_slot1=44 會對到好幾條掃描線上的所有樣本
            specs = parse_holdout(args.holdout, design_params)
            held = [r for r in runs if any(abs(r["params"][k] - v) < 1e-9 for k, v in specs)]
        else:
            held = default_holdout(runs, design_params)
        held_ids = {id(r) for r in held}
        train = [r for r in runs if id(r) not in held_ids]

        if held and train:
            held_desc = [f"{r.get('source', '')}: " + format_params(design_params, [r["params"][k] for k in design_params])
                         for r in held]
            print(f"=== 留一驗證：隱藏 {len(held)} 組樣本 {held_desc}，僅用其餘 {len(train)} 組訓練 ===")
            models_v, norm_v = train_ensemble(train, design_params, bands,
                                              epochs=args.epochs, n_ensemble=args.n_ensemble, verbose=True)

            fig, axes = plt.subplots(len(held), 1, figsize=(10, 3.8 * len(held)), squeeze=False)
            for ax, r in zip(axes[:, 0], held):
                pv = [r["params"][k] for k in design_params]
                pred_mean, pred_std = predict_single(models_v, norm_v, pv, r["freq"])
                rmse = np.sqrt(np.mean((pred_mean - r["s11"]) ** 2))
                mt = band_metrics(r["freq"], r["s11"], bands)
                mp = band_metrics(r["freq"], pred_mean, bands)
                p_str = format_params(design_params, pv)
                print(f"  驗證 [{p_str}]: 全頻 RMSE = {rmse:.2f} dB | {args.band} 覆蓋率 "
                      f"真實 {mt['coverage'] * 100:.1f}% vs 預測 {mp['coverage'] * 100:.1f}% "
                      f"(平均不確定性 ±{np.mean(pred_std):.2f} dB)")

                ax.plot(r["freq"], r["s11"], "k-", lw=1.6, label="CST 模擬 (真實)")
                ax.plot(r["freq"], pred_mean, "r--", lw=1.4, label="AI 集成預測 (平均)")
                ax.fill_between(r["freq"], pred_mean - 2 * pred_std, pred_mean + 2 * pred_std,
                                color="red", alpha=0.18, label="AI 95% 信心區間 (±2σ)")
                for lo, hi in bands:
                    ax.axvspan(lo, hi, color="tab:green", alpha=0.1)
                ax.axhline(S11_GOAL_DB, color="gray", ls=":")
                ax.set_title(f"留一驗證 [{p_str}]  RMSE={rmse:.2f} dB")
                ax.set_xlabel("Frequency (GHz)")
                ax.set_ylabel("S11 (dB)")
                ax.legend(loc="lower left")
                ax.grid(alpha=0.3)

            fig.tight_layout()
            val_path = os.path.join(OUT_DIR, "validation.png")
            fig.savefig(val_path, dpi=140)
            plt.close(fig)
            print(f"  驗證圖 (含信心區間陰影帶) 已存：{val_path}")

    # ---- 使用全部資料訓練最終集成模型 ----
    print(f"=== 使用全部 {len(runs)} 組資料訓練最終代理模型 (Ensemble={args.n_ensemble}) ===")
    models, norm = train_ensemble(runs, design_params, bands,
                                  epochs=args.epochs, n_ensemble=args.n_ensemble, verbose=True)
    torch.save({
        "state_dicts": [m.state_dict() for m in models],
        "norm": norm.state(),
        "design_params": design_params,
        "fixed_params": fixed,
    }, MODEL_PATH)
    print(f"  最終模型權重已存：{MODEL_PATH}")

    # ---- 用代理模型細掃單/多參數設計空間 ----
    sweeps_path = os.path.join(OUT_DIR, "sweeps.png")
    plot_sweeps(runs, design_params, models, norm, bands, sweeps_path)
    print(f"  各掃描線 CST vs 代理模型熱圖已存：{sweeps_path}")

    param_grid, grid_axes = generate_search_grid(runs, design_params, trust_radius=args.trust_radius)
    fine_freq = np.linspace(freq[0], freq[-1], len(freq))
    mean_curves, std_curves = predict_curves_batch(models, norm, param_grid, fine_freq)

    metrics = [band_metrics(fine_freq, c, bands) for c in mean_curves]
    scores = np.array([score(m) for m in metrics])
    uncertainties = np.mean(std_curves, axis=1)  # 每組設計在全頻段的平均標準差

    order = np.argsort(-scores)

    print(f"=== 代理模型細掃設計空間 {design_params} (共搜尋 {len(param_grid)} 組參數組合) ===")
    if len(design_params) >= 4:
        print(f"  (只搜尋離 CST 樣本 {args.trust_radius:.0%} 參數範圍以內的組合；遠離資料的組合模型沒有依據)")
    print(f"目標：{args.band} 頻段 S11 < {S11_GOAL_DB} dB 的均衡覆蓋率最大")
    for rank, i in enumerate(order[:5], 1):
        m = metrics[i]
        p_str = format_params(design_params, param_grid[i])
        print(f"  第{rank}名 [{p_str}]  均衡覆蓋率={m['coverage'] * 100:5.1f}%  "
              f"平均={m['mean_db']:6.2f} dB  最低={m['best_db']:6.2f} dB  (不確定性 ±{uncertainties[i]:.2f} dB)")

    best_idx = order[0]
    best_params = param_grid[best_idx]
    best_curve = mean_curves[best_idx]
    best_std = std_curves[best_idx]
    best_p_str = format_params(design_params, best_params)

    print(f"★ 建議最佳設計參數：[{best_p_str}]")
    print("  預測的 -10 dB 頻寬區段：")
    for fl, fh, fm, sm in resonances(fine_freq, best_curve):
        print(f"    {fl:.3f}-{fh:.3f} GHz (BW {(fh - fl) * 1000:.0f} MHz)，共振 {fm:.3f} GHz @ {sm:.1f} dB")

    # ---- 主動學習 (Active Learning / UCB) 推薦下一次 CST 模擬點 ----
    # 挑出「分數高」且「模型不確定性較大」的點，幫助下一輪迭代最有效率地補齊模型盲區
    acq_scores = scores + 15.0 * uncertainties
    al_idx = int(np.argmax(acq_scores))
    if al_idx != best_idx:
        al_p_str = format_params(design_params, param_grid[al_idx])
        print(f"💡 [主動學習推薦] 若要補跑 CST 來提升模型準確度，建議優先加跑：[{al_p_str}] "
              f"(預期覆蓋率 {metrics[al_idx]['coverage']*100:.1f}%，模型不確定性 ±{uncertainties[al_idx]:.2f} dB)")

    # ---- 繪圖：自動根據 1D 或 多維參數 調整視覺化圖表 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9.5))

    if len(design_params) == 1:
        # 1D 參數：畫 Frequency vs Parameter 的 S11 熱圖
        dp = design_params[0]
        sweep_1d = param_grid[:, 0]
        im = ax1.imshow(mean_curves, aspect="auto", origin="lower", cmap="viridis_r", vmin=-25, vmax=0,
                        extent=[fine_freq[0], fine_freq[-1], sweep_1d[0], sweep_1d[-1]])
        for lo, hi in bands:
            ax1.axvline(lo, color="w", ls="--", lw=0.8)
            ax1.axvline(hi, color="w", ls="--", lw=0.8)
        ax1.axhline(best_params[0], color="r", lw=1.2)
        ax1.set_xlabel("Frequency (GHz)")
        ax1.set_ylabel(dp)
        ax1.set_title(f"代理模型預測 S11 (dB) vs {dp}（紅線 = 最佳建議值 {best_params[0]:.2f}）")
        fig.colorbar(im, ax=ax1, label="S11 (dB)")
    elif len(design_params) == 2 and grid_axes is not None:
        # 2D 溝槽參數：畫 Parameter 1 vs Parameter 2 的「目標頻段覆蓋率」2D 決策熱圖
        p0_name, p1_name = design_params[0], design_params[1]
        ax0, ax1_grid = grid_axes[0], grid_axes[1]
        cov_map = np.array([m["coverage"] * 100.0 for m in metrics]).reshape(len(ax0), len(ax1_grid)).T
        im = ax1.imshow(cov_map, aspect="auto", origin="lower", cmap="YlGnBu", vmin=0, vmax=100,
                        extent=[ax0[0], ax0[-1], ax1_grid[0], ax1_grid[-1]])
        # 標出原始 CST 取樣點與 AI 最佳點
        raw_p0 = [r["params"][p0_name] for r in runs]
        raw_p1 = [r["params"][p1_name] for r in runs]
        ax1.scatter(raw_p0, raw_p1, c="black", marker="o", s=35, label="原始 CST 模擬點")
        ax1.scatter([best_params[0]], [best_params[1]], c="red", marker="*", s=180, label=f"AI 最佳點 ({best_p_str})")
        ax1.set_xlabel(p0_name)
        ax1.set_ylabel(p1_name)
        ax1.set_title(f"雙參數設計空間決策熱圖：{args.band} 頻段覆蓋率 (%)")
        ax1.legend(loc="upper right")
        fig.colorbar(im, ax=ax1, label="Band Coverage (%)")
    else:
        # 3D 以上參數：畫 Top 15 參數組合的綜合評分與覆蓋率長條圖
        top_k = min(12, len(order))
        top_indices = order[:top_k][::-1]
        labels = [format_params(design_params, param_grid[i]) for i in top_indices]
        covs = [metrics[i]["coverage"] * 100.0 for i in top_indices]
        ax1.barh(range(top_k), covs, color="tab:blue", alpha=0.8)
        ax1.set_yticks(range(top_k))
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel(f"{args.band} 頻段均衡覆蓋率 (%)")
        ax1.set_title(f"多參數搜尋 Top {top_k} 溝槽尺寸組合排行")
        ax1.grid(axis="x", alpha=0.3)

    # 下方子圖：最佳參數的 S11 曲線 + 95% 信心區間 + 最接近的 CST 樣本對比
    ax2.plot(fine_freq, best_curve, "r-", lw=1.6, label=f"AI 最佳預測 [{best_p_str}]")
    ax2.fill_between(fine_freq, best_curve - 2 * best_std, best_curve + 2 * best_std,
                     color="red", alpha=0.18, label="AI 95% 信心區間 (±2σ)")

    # 找歐氏距離最接近的原始 CST 樣本
    def dist_to_best(r):
        rv = norm.p([r["params"][k] for k in design_params])
        bv = norm.p(best_params)
        return float(np.linalg.norm(rv - bv))

    nearest = min(runs, key=dist_to_best)
    near_str = format_params(design_params, [nearest["params"][k] for k in design_params])
    ax2.plot(nearest["freq"], nearest["s11"], "k--", lw=1.2,
             label=f"最接近的 CST 樣本 [{near_str}]")

    for lo, hi in bands:
        ax2.axvspan(lo, hi, color="tab:green", alpha=0.12)
    ax2.axhline(S11_GOAL_DB, color="gray", ls=":")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("S11 (dB)")
    ax2.set_title(f"{args.band} 最佳化結果與信心區間")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    opt_path = os.path.join(OUT_DIR, f"optimize_{args.band}.png")
    fig.savefig(opt_path, dpi=140)
    plt.close(fig)
    print(f"最佳化視覺圖表已存：{opt_path}")


if __name__ == "__main__":
    main()
