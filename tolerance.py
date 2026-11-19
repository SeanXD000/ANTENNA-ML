"""
製程容差分析（Monte Carlo）
------------------------------------------------
用法：
  python tolerance.py data/verify_best3.txt
  python tolerance.py data/verify_best3.txt --tol 0.05 0.1 0.2 --n 2000 --band 6E

做法：
  以 CST 實測過的設計為名目點，每個設計變數各自加上均勻分布 ±tol (mm) 的隨機誤差，
  用代理模型估計誤差後的 S11，統計目標頻段覆蓋率掉到 100% 以下的機率（良率）。

  模型在名目點本身就有 1–2 dB 的誤差，所以不直接用模型曲線，而是
      估計 S11(p) = CST 實測 S11(名目) + [模型(p) − 模型(名目)]
  只借用模型「參數變多少、S11 跟著變多少」的局部變化，絕對值以 CST 實測為準。
  限制：只含設計變數（W_slot1/2/3、Ymax）的誤差，不含板厚、介電常數、銅厚等固定參數；
  擾動很小時模型差值的可信度較高，±0.5 mm 以上要保守看待。

輸出：終端機表格、outputs/tolerance_<檔名>.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from antenna_ml import (MODEL_PATH, OUT_DIR, S11_GOAL_DB, WIFI_BANDS, band_metrics, format_params,
                        parse_cst_sweep, predict_curves_batch)
from iterative_train import load_checkpoint


def estimate(models, norm, nominal, s11_cst, freq, grid):
    """估計 grid 每組參數的 S11：CST 實測 + 模型差值。回傳 (曲線, 模型差值的集成標準差平均)。"""
    m0, _ = predict_curves_batch(models, norm, [nominal], freq)
    mc, sd = predict_curves_batch(models, norm, grid, freq)
    return s11_cst[None, :] + (mc - m0), sd.mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description="製程容差分析（Monte Carlo）")
    ap.add_argument("file", help="CST 實測的名目設計（如 data/verify_best3.txt），只取第一組")
    ap.add_argument("--tol", type=float, nargs="+", default=[0.05, 0.1, 0.2],
                    help="每個設計變數的誤差範圍 ±tol (mm)，可給多個")
    ap.add_argument("--n", type=int, default=2000, help="每個 tol 的隨機樣本數")
    ap.add_argument("--band", default="6E", choices=list(WIFI_BANDS))
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()
    bands = WIFI_BANDS[args.band]

    models, norm, design_params = load_checkpoint(args.model)
    run = parse_cst_sweep(args.file)[0]
    freq, s11 = run["freq"], run["s11"]
    nominal = np.array([run["params"][k] for k in design_params], dtype=float)
    name = os.path.splitext(os.path.basename(args.file))[0]
    m_nom = band_metrics(freq, s11, bands)
    print(f"名目設計 {name}：[{format_params(design_params, nominal)}]")
    print(f"  CST 實測 {args.band}：覆蓋率 {m_nom['coverage'] * 100:.1f}%、帶內最差 {m_nom['worst_db']:.2f} dB、"
          f"邊界餘裕 {m_nom['margin_db']:.2f} dB")

    # ---- 1. 單一參數敏感度：其他參數固定，只動一個 ----
    offs = np.linspace(-0.5, 0.5, 21)
    sens = {}
    print(f"\n=== 單一參數敏感度（帶內最差 S11，越負越好；−10 dB 以上就掉覆蓋率）===")
    print(f"  {'參數':<10}{'-0.2 mm':>10}{'名目':>10}{'+0.2 mm':>10}{'斜率 (dB/mm)':>15}")
    for j, k in enumerate(design_params):
        grid = np.tile(nominal, (len(offs), 1))
        grid[:, j] += offs
        est, _ = estimate(models, norm, nominal, s11, freq, grid)
        worst = np.array([band_metrics(freq, c, bands)["worst_db"] for c in est])
        sens[k] = worst
        near = np.abs(offs) <= 0.2 + 1e-9
        slope = np.polyfit(offs[near], worst[near], 1)[0]
        i_m, i_0, i_p = np.argmin(np.abs(offs + 0.2)), np.argmin(np.abs(offs)), np.argmin(np.abs(offs - 0.2))
        print(f"  {k:<10}{worst[i_m]:>10.2f}{worst[i_0]:>10.2f}{worst[i_p]:>10.2f}{slope:>15.2f}")

    # ---- 2. Monte Carlo：所有參數同時加隨機誤差 ----
    rng = np.random.default_rng(0)
    mc_results = {}
    print(f"\n=== Monte Carlo（每個參數各自均勻分布 ±tol，{args.n} 組）===")
    print(f"  {'±tol (mm)':<11}{'良率 (覆蓋率100%)':>18}{'覆蓋率 最差':>12}{'覆蓋率 5%':>11}"
          f"{'最差 S11 中位':>13}{'最差 S11 95%':>13}{'模型差值 σ':>11}")
    for tol in args.tol:
        grid = nominal + rng.uniform(-tol, tol, size=(args.n, len(design_params)))
        est, sd = estimate(models, norm, nominal, s11, freq, grid)
        met = [band_metrics(freq, c, bands) for c in est]
        cov = np.array([m["coverage"] for m in met]) * 100
        worst = np.array([m["worst_db"] for m in met])
        mc_results[tol] = (cov, worst, est)
        print(f"  ±{tol:<10g}{np.mean(cov >= 100 - 1e-9) * 100:>17.1f}%{cov.min():>11.1f}%"
              f"{np.percentile(cov, 5):>10.1f}%{np.median(worst):>13.2f}{np.percentile(worst, 95):>13.2f}"
              f"{sd.mean():>11.2f}")
    print("  （最差 S11 95% = 95% 的樣本帶內最差點都比這個值好；> −10 dB 代表有樣本掉覆蓋率）")
    print("  （模型差值 σ = 集成模型對這些樣本的平均不確定性，差值本身的誤差通常比它小，僅供參考）")

    # ---- 3. 畫圖 ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    for k in design_params:
        ax.plot(offs, sens[k], marker="o", ms=3, label=k)
    ax.axhline(S11_GOAL_DB, color="tab:red", ls=":", lw=1)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("參數偏移 (mm)")
    ax.set_ylabel(f"{args.band} 帶內最差 S11 (dB)")
    ax.set_title("單一參數敏感度")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    bins = np.linspace(min(w.min() for _, w, _ in mc_results.values()) - 0.2, -9.0, 50)
    for tol, (_, worst, _) in mc_results.items():
        ax.hist(worst, bins=bins, alpha=0.45, label=f"±{tol:g} mm")
    ax.axvline(S11_GOAL_DB, color="tab:red", ls=":", lw=1)
    ax.axvline(m_nom["worst_db"], color="k", lw=1, label="名目（CST）")
    ax.set_xlabel(f"{args.band} 帶內最差 S11 (dB)")
    ax.set_ylabel("樣本數")
    ax.set_title("Monte Carlo：帶內最差 S11 分布")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[2]
    tol_show = max(args.tol)
    est = mc_results[tol_show][2]
    lo, hi = np.percentile(est, 5, axis=0), np.percentile(est, 95, axis=0)
    ax.fill_between(freq, lo, hi, color="tab:blue", alpha=0.25, label=f"±{tol_show:g} mm 的 5–95% 範圍")
    ax.plot(freq, s11, "k-", lw=1.5, label="名目（CST 實測）")
    for lo_b, hi_b in bands:
        ax.axvspan(lo_b, hi_b, color="tab:green", alpha=0.1)
    ax.axhline(S11_GOAL_DB, color="tab:red", ls=":", lw=1)
    ax.set_xlim(5.0, 8.0)
    ax.set_xlabel("頻率 (GHz)")
    ax.set_ylabel("S11 (dB)")
    ax.set_title("S11 變動範圍")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    fig.suptitle(f"製程容差分析：{name} [{format_params(design_params, nominal)}]", fontsize=11)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"tolerance_{name}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n圖：{out}")


if __name__ == "__main__":
    main()
