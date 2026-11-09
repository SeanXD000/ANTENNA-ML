"""
機器學習輔助 WiFi 6E/7 天線設計最佳化
------------------------------------------------
[整合說明] 本檔 = 舊版 antenna_ml_r2bygemini.py (V2)，它完整涵蓋舊 antenna_ml.py (V1) 的功能，
          V1 已移除（備份在 ../antenna-ml_backup_20261110/）。詳見 README.md「整合紀錄」。

流程：
 1. 讀取 CST 參數掃描匯出的 .txt（一個檔案含多組 #Parameters 區塊）
 2. 自動找出有變動的設計參數（例如 W4）當作模型輸入
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

用法：
  python antenna_ml.py                         # 驗證 + 訓練 + 最佳化 (自動支援單/多參數)
  python antenna_ml.py --band WIFI7            # 目標頻段: 6E / 5G / 2G4 / WIFI7
  python antenna_ml.py --holdout 6 10          # 指定驗證時要藏起來的參數值（第一個設計變數）
  python antenna_ml.py --n-ensemble 5          # 使用 5 個模型集成並繪製不確定性陰影帶
  python antenna_ml.py --skip-validate         # 直接用全部資料訓練
"""
import argparse
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


def load_all_runs(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")) + glob.glob(os.path.join(data_dir, "*.dat")))
    if not files:
        raise FileNotFoundError(f"在 '{data_dir}' 找不到任何 .txt / .dat 檔案")
    runs = []
    for fp in files:
        rs = parse_cst_sweep(fp)
        print(f"讀取 {os.path.basename(fp)}：{len(rs)} 組模擬")
        runs.extend(rs)
    if not runs:
        raise ValueError("讀取到的模擬資料為空，請確認 CST 匯出檔格式是否包含 #Parameters {...}")

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


def train_ensemble(runs, design_params, bands, epochs=2500, n_ensemble=5, verbose=True):
    """訓練多個代理模型組成 Ensemble，用於預測平均值與不確定性（標準差）。"""
    P, F, Y = runs_to_arrays(runs, design_params)
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


def generate_search_grid(runs, design_params, max_total_points=5000):
    """自動根據參數數量建立搜尋網格（支援 1D 細掃、2D/3D 網格、高維度隨機取樣）。"""
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
        # 4 個參數以上使用均勻隨機/準蒙地卡羅抽樣避免維度爆炸
        rng = np.random.default_rng(42)
        combos = np.zeros((max_total_points, n_p), dtype=float)
        for idx, (lo, hi) in enumerate(bounds):
            combos[:, idx] = np.round(rng.uniform(lo, hi, size=max_total_points), 3)
        return combos, None


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
    ap.add_argument("--holdout", type=float, nargs="*", default=None,
                    help="指定第一個設計參數要隱藏驗證的數值（若未指定，自動挑中間 2 組樣本）")
    ap.add_argument("--epochs", type=int, default=2500, help="訓練 Epochs 數")
    ap.add_argument("--n-ensemble", type=int, default=5, help="集成模型數量 (計算信心區間)")
    ap.add_argument("--skip-validate", action="store_true", help="跳過留一驗證")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    bands = WIFI_BANDS[args.band]

    print(f"使用運算裝置：{DEVICE}")
    runs, design_params, fixed = load_all_runs(args.data)
    freq = runs[0]["freq"]
    print(f"共 {len(runs)} 組模擬，每組 {len(freq)} 個頻點 ({freq[0]:.2f}-{freq[-1]:.2f} GHz)")
    print(f"自動偵測到的設計變數 (共 {len(design_params)} 個)：{design_params}")
    print(f"固定不變的參數：{fixed}")

    # 印出各設計變數的取樣範圍
    for k in design_params:
        u_vals = sorted({r["params"][k] for r in runs})
        print(f"  參數 [{k}] 取樣值 ({len(u_vals)} 種): {u_vals}")
    print()

    # ---- 原始模擬資料在目標頻段的表現 ----
    print(f"=== 原始 CST 資料在 {args.band} 頻段 {bands} 的表現 (前 10 筆) ===")
    sorted_runs = sorted(runs, key=lambda r: [r["params"][k] for k in design_params])
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
        dp0 = design_params[0]
        if args.holdout:
            held = [r for r in runs if r["params"][dp0] in args.holdout]
            train = [r for r in runs if r["params"][dp0] not in args.holdout]
        else:
            # 自動挑選排序後位於 1/3 與 2/3 位置的 2 組資料作為內部驗證集（避免選到邊界外插）
            idx1 = len(sorted_runs) // 3
            idx2 = (len(sorted_runs) * 2) // 3
            held_indices = {idx1, idx2} if idx1 != idx2 else {idx1}
            held = [sorted_runs[i] for i in sorted(held_indices)]
            train = [sorted_runs[i] for i in range(len(sorted_runs)) if i not in held_indices]

        if held and train:
            held_desc = [format_params(design_params, [r["params"][k] for k in design_params]) for r in held]
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
    param_grid, grid_axes = generate_search_grid(runs, design_params)
    fine_freq = np.linspace(freq[0], freq[-1], len(freq))
    mean_curves, std_curves = predict_curves_batch(models, norm, param_grid, fine_freq)

    metrics = [band_metrics(fine_freq, c, bands) for c in mean_curves]
    scores = np.array([score(m) for m in metrics])
    uncertainties = np.mean(std_curves, axis=1)  # 每組設計在全頻段的平均標準差

    order = np.argsort(-scores)

    print(f"=== 代理模型細掃設計空間 {design_params} (共搜尋 {len(param_grid)} 組參數組合) ===")
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
