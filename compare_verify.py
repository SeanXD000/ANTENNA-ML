"""
CST 實測 vs 模型預測 自動比對
------------------------------------------------
用法：
  python compare_verify.py data/verify_best3.txt
  python compare_verify.py data/verify_best3.txt --band 6E --note "重訓後第二個建議點"

流程：
 1. 讀取 CST 匯出的驗證檔（格式同 data/ 內其他檔，可含多組 #Parameters）
 2. 載入 antenna_surrogate.pth，在同樣的參數與頻點上預測 S11 與不確定性
 3. 印出比對結果：全頻／帶內 RMSE、覆蓋率、平均 S11、-10 dB 頻寬邊界、共振頻率、不確定性校準
 4. 追加一列到 outputs/verify_log.csv（Excel 可直接開），並畫圖存成 outputs/compare_<檔名>.png

注意：要在「重訓之前」執行。模型若已經用這個檔訓練過，比對結果會失真（程式會依檔案時間提醒）。
"""
import argparse
import csv
import datetime
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from antenna_ml import (ACC_COV_PP, ACC_EDGE_GHZ, CAL_NEAR_DB, EDGE_PAD_GHZ, MODEL_PATH, OUT_DIR,
                        S11_GOAL_DB, WIFI_BANDS, band_metrics, design_accuracy, format_params, in_band_mask,
                        main_segment, parse_cst_sweep, predict_single, print_accuracy_summary)
from iterative_train import load_checkpoint

LOG_PATH = os.path.join(OUT_DIR, "verify_log.csv")
LOG_FIELDS = [
    "time", "file", "run", "band", "params", "in_train_range",
    "rmse_full_db", "rmse_band_db",
    "cov_true", "cov_pred", "cov_err_pp",
    "mean_db_true", "mean_db_pred",
    "edge_lo_true", "edge_lo_pred", "edge_hi_true", "edge_hi_pred",
    "res_f_true", "res_f_pred", "res_db_true", "res_db_pred",
    "sigma_full_db", "sigma_band_db", "calib_ratio", "within_2sigma",
    "model_mtime", "note",
]


def fmt(x, spec=".3f"):
    return "—" if x is None else format(x, spec)


def compare_run(run, models, norm, design_params, fixed, bands):
    freq, s11 = run["freq"], run["s11"]
    missing = [k for k in design_params if k not in run["params"]]
    if missing:
        raise SystemExit(f"驗證檔缺少設計變數 {missing}，無法比對")
    pv = np.array([run["params"][k] for k in design_params], dtype=float)

    # 固定參數不同 = 不是同一個天線，比對沒有意義
    diff_fixed = {k: (v, run["params"].get(k)) for k, v in (fixed or {}).items()
                  if run["params"].get(k) is not None and abs(run["params"][k] - v) > 1e-9}
    if diff_fixed:
        print(f"  [注意]固定參數與訓練資料不同（訓練值, 驗證值）：{diff_fixed}")

    # 參數是否在訓練範圍內（超出 = 外推）
    out_of_range = [k for k, v, lo, hi in zip(design_params, pv, norm.p_min, norm.p_max)
                    if v < lo - 1e-9 or v > hi + 1e-9]

    pred, std = predict_single(models, norm, pv, freq)
    mb = in_band_mask(freq, bands)
    err = pred - s11

    m_true, m_pred = band_metrics(freq, s11, bands), band_metrics(freq, pred, bands)
    seg_t, seg_p = main_segment(freq, s11, bands), main_segment(freq, pred, bands)

    rmse_full = float(np.sqrt(np.mean(err ** 2)))
    rmse_band = float(np.sqrt(np.mean(err[mb] ** 2)))
    sigma_full = float(np.mean(std))
    sigma_band = float(np.mean(std[mb]))
    within = float(np.mean(np.abs(err) <= 2 * std))
    # -10 dB 附近（頻段 ±EDGE_PAD_GHZ 內、實測或預測在 CAL_NEAR_DB）：模型的 σ 是依這些頻點校正的
    near = lambda s: (s >= CAL_NEAR_DB[0]) & (s <= CAL_NEAR_DB[1])
    mn = in_band_mask(freq, bands, EDGE_PAD_GHZ) & (near(s11) | near(pred))
    ratio_near = (float(np.sqrt(np.mean(err[mn] ** 2)) / np.mean(std[mn]))
                  if mn.sum() >= 5 and np.mean(std[mn]) > 1e-9 else None)

    return dict(
        params=format_params(design_params, pv),
        in_train_range="yes" if not out_of_range else "no:" + "/".join(out_of_range),
        rmse_full_db=rmse_full, rmse_band_db=rmse_band,
        cov_true=m_true["coverage"] * 100, cov_pred=m_pred["coverage"] * 100,
        cov_err_pp=(m_pred["coverage"] - m_true["coverage"]) * 100,
        mean_db_true=m_true["mean_db"], mean_db_pred=m_pred["mean_db"],
        edge_lo_true=seg_t[0] if seg_t else None, edge_lo_pred=seg_p[0] if seg_p else None,
        edge_hi_true=seg_t[1] if seg_t else None, edge_hi_pred=seg_p[1] if seg_p else None,
        res_f_true=seg_t[2] if seg_t else None, res_f_pred=seg_p[2] if seg_p else None,
        res_db_true=seg_t[3] if seg_t else None, res_db_pred=seg_p[3] if seg_p else None,
        sigma_full_db=sigma_full, sigma_band_db=sigma_band,
        calib_ratio=rmse_full / sigma_full if sigma_full > 1e-9 else None,
        within_2sigma=within * 100,
        _acc=design_accuracy(freq, s11, pred, bands), _ratio_near=ratio_near, _n_near=int(mn.sum()),
        _freq=freq, _s11=s11, _pred=pred, _std=std,
    )


def print_report(r, band_name):
    d = lambda a, b: None if a is None or b is None else b - a
    print(f"  參數：{r['params']}")
    if r["in_train_range"] != "yes":
        print(f"  [注意]超出訓練資料範圍（外推）：{r['in_train_range'][3:]}")
    print(f"  {'指標':<22}{'CST 實測':>12}{'模型預測':>12}{'差 (預測-實測)':>16}")
    rows = [
        (f"{band_name} 覆蓋率 (%)", r["cov_true"], r["cov_pred"], ".1f"),
        (f"{band_name} 平均 S11 (dB)", r["mean_db_true"], r["mean_db_pred"], ".2f"),
        ("-10 dB 下緣 (GHz)", r["edge_lo_true"], r["edge_lo_pred"], ".3f"),
        ("-10 dB 上緣 (GHz)", r["edge_hi_true"], r["edge_hi_pred"], ".3f"),
        ("共振頻率 (GHz)", r["res_f_true"], r["res_f_pred"], ".3f"),
        ("共振深度 (dB)", r["res_db_true"], r["res_db_pred"], ".2f"),
    ]
    for name, t, p, spec in rows:
        print(f"  {name:<22}{fmt(t, spec):>12}{fmt(p, spec):>12}{fmt(d(t, p), '+' + spec):>16}")
    print(f"  RMSE：全頻 {r['rmse_full_db']:.2f} dB｜帶內 {r['rmse_band_db']:.2f} dB")
    ratio = r["calib_ratio"]
    print(f"  不確定性：模型自報平均 ±{r['sigma_full_db']:.2f} dB（帶內 ±{r['sigma_band_db']:.2f}）"
          f"｜實際誤差 / 自報 = {fmt(ratio, '.1f')} 倍｜誤差落在 ±2σ 內的頻點 {r['within_2sigma']:.0f}%")
    rn = r["_ratio_near"]
    print(f"  -10 dB 附近（{r['_n_near']} 個頻點，σ 依這些頻點校正）：實際誤差 / 自報 = {fmt(rn, '.1f')} 倍")
    if rn is not None and rn > 2:
        print("  [注意]模型在 -10 dB 附近低估了不確定性（實際誤差超過自報的 2 倍）")
    a = r["_acc"]
    e_str = ("兩邊都沒有 -10 dB 區段" if a["edge_lo_err"] is None else "只有一邊有 -10 dB 區段"
             if np.isinf(a["edge_lo_err"]) else
             f"邊緣誤差 下 {a['edge_lo_err'] * 1000:+.0f} / 上 {a['edge_hi_err'] * 1000:+.0f} MHz")
    print(f"  設計用途判定（覆蓋率誤差 ≤ {ACC_COV_PP:g} 個百分點、邊緣 ≤ {ACC_EDGE_GHZ * 1000:.0f} MHz）："
          f"覆蓋率 {a['cov_err_pp']:+.1f}｜{e_str} → {'通過' if a['ok'] else '未通過'}")


def plot_run(r, bands, title, out_path):
    f, s, p, sd = r["_freq"], r["_s11"], r["_pred"], r["_std"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for lo, hi in bands:
        ax.axvspan(lo, hi, color="tab:green", alpha=0.10)
    ax.fill_between(f, p - 2 * sd, p + 2 * sd, color="tab:orange", alpha=0.3, label="模型 ±2σ")
    ax.plot(f, s, "k-", lw=1.8, label="CST 實測")
    ax.plot(f, p, "--", color="tab:orange", lw=1.6, label="模型預測")
    ax.axhline(S11_GOAL_DB, color="tab:red", ls=":", lw=1)
    ax.set_xlabel("頻率 (GHz)")
    ax.set_ylabel("S11 (dB)")
    ax.set_title(f"{title}\n{r['params']}\n全頻 RMSE {r['rmse_full_db']:.2f} dB｜帶內 {r['rmse_band_db']:.2f} dB"
                 f"｜覆蓋率 實測 {r['cov_true']:.1f}% / 預測 {r['cov_pred']:.1f}%", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def append_log(row):
    os.makedirs(OUT_DIR, exist_ok=True)
    new = not os.path.exists(LOG_PATH)
    # utf-8-sig：讓 Excel 正確顯示中文
    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: (round(v, 4) if isinstance(v, float) else ("" if v is None else v))
                    for k, v in row.items() if k in LOG_FIELDS})


def main():
    ap = argparse.ArgumentParser(description="CST 驗證結果 vs 模型預測比對")
    ap.add_argument("files", nargs="+", help="CST 匯出的驗證檔（如 data/verify_best3.txt）")
    ap.add_argument("--band", default="6E", choices=list(WIFI_BANDS), help="目標頻段")
    ap.add_argument("--model", default=MODEL_PATH, help="模型檔路徑")
    ap.add_argument("--note", default="", help="寫進紀錄檔的備註")
    ap.add_argument("--no-log", action="store_true", help="不寫入 verify_log.csv（只看結果）")
    args = ap.parse_args()
    bands = WIFI_BANDS[args.band]

    if not os.path.exists(args.model):
        raise SystemExit(f"找不到模型 {args.model}，請先執行 antenna_ml.py 訓練")
    models, norm, design_params = load_checkpoint(args.model)
    import torch
    fixed = torch.load(args.model, map_location="cpu", weights_only=False).get("fixed_params")
    model_mtime = os.path.getmtime(args.model)
    model_time_str = datetime.datetime.fromtimestamp(model_mtime).strftime("%Y-%m-%d %H:%M")
    print(f"載入模型 {args.model}（{len(models)} 個集成成員，存檔時間 {model_time_str}）")

    results = []
    for fp in args.files:
        runs = parse_cst_sweep(fp)
        name = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n=== {os.path.basename(fp)}：{len(runs)} 組模擬 ===")
        if not runs:
            print("  讀不到任何模擬資料，略過")
            continue
        if os.path.getmtime(fp) < model_mtime:
            print(f"  [注意]模型存檔時間比這個檔新，模型可能已經用它訓練過，比對結果會偏樂觀。"
                  f"\n    正確做法：拿到 CST 結果後、重訓之前先跑這支程式。")

        for i, run in enumerate(runs, 1):
            if len(runs) > 1:
                print(f"\n  --- 第 {i} 組 ---")
            r = compare_run(run, models, norm, design_params, fixed, bands)
            results.append(r)
            print_report(r, args.band)
            suffix = f"_{i}" if len(runs) > 1 else ""
            out_png = os.path.join(OUT_DIR, f"compare_{name}{suffix}.png")
            os.makedirs(OUT_DIR, exist_ok=True)
            plot_run(r, bands, f"{name}{suffix}：CST 實測 vs 模型預測（{args.band}）", out_png)
            print(f"  圖：{out_png}")
            if not args.no_log:
                append_log(dict(r, time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                file=os.path.basename(fp), run=i, band=args.band,
                                model_mtime=model_time_str, note=args.note))

    if len(results) > 1:
        print(f"\n=== 總結（{len(results)} 組）===")
        print(f"  RMSE 平均：全頻 {np.mean([r['rmse_full_db'] for r in results]):.2f} dB｜"
              f"帶內 {np.mean([r['rmse_band_db'] for r in results]):.2f} dB")
        rn = [r["_ratio_near"] for r in results if r["_ratio_near"] is not None]
        if rn:
            print(f"  -10 dB 附近 實際誤差 / 自報 σ：中位數 {np.median(rn):.1f} 倍（1 倍 = 校準良好）")
        print_accuracy_summary([(True, r["_acc"]) for r in results])
        k, n = sum(r["_acc"]["ok"] for r in results), len(results)
        if k == n:  # 全部通過時的單邊 95% 信賴下限（Clopper-Pearson）：0.05^(1/n)
            print(f"    全部通過：真實通過率的 95% 信賴下限約 {0.05 ** (1 / n) * 100:.0f}%（樣本數 {n}）")

    if not args.no_log:
        print(f"\n紀錄已追加到 {LOG_PATH}")


if __name__ == "__main__":
    main()
