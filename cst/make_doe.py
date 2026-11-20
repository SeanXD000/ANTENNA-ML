"""產生多參數聯合取樣清單（拉丁超立方 LHS，挑出點與點之間最分散的一組）。

用法：
  python cst/make_doe.py --batch 2          # 產生 cst/doe_batch2.csv 與 cst/run_doe_batch2.bas
  python cst/make_doe.py --batch 1          # 重新產生第一批（結果與原本相同）
  python cst/make_doe.py --test             # 獨立測試集：cst/test_set.csv 與 cst/run_test_set.bas

輸出：
  cst/doe_batch<N>.csv       參數清單，可手動輸入 CST
  cst/run_doe_batch<N>.bas   CST macro（第 1 批是 cst/run_doe.bas），以 run_doe.bas 為範本，
                              只換掉參數表、組數與輸出檔名；跑完會寫到 data/doe_batch<N>.txt
  --test 的 macro 寫到 data_test/test_set.txt：不在 data/ 裡，antenna_ml.py / iterative_train.py 不會拿去訓練，
  只用 compare_verify.py 評估「定版模型」。測試集一旦拿去訓練就不再獨立，要測新模型得另取一批（換 seed）。
"""
import argparse
import csv
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_BAS = os.path.join(HERE, "run_doe.bas")  # 第 1 批的 macro，當作範本

BATCHES = {
    # 第 1 批：以當時最佳點 W_slot1=44, W_slot2=14, W_slot3=22, Ymax=18.5 為中心
    1: dict(ranges={"W_slot1": (40.0, 45.0), "W_slot2": (12.0, 18.0),
                    "W_slot3": (18.0, 26.0), "Ymax": (14.0, 18.5)}, n=25, seed=2026),
    # 第 2 批：模型建議的最佳區域在 W_slot1 38–39、W_slot2 12–14、W_slot3 23 左右、Ymax 13–15，
    # 已落在第 1 批範圍外（verify_best2 是唯一輸給內插 baseline 的驗證點）。
    # W_slot1 往下延伸到 36、和第 1 批重疊 40–42 以銜接；Ymax 往下延伸到 11。
    2: dict(ranges={"W_slot1": (36.0, 42.0), "W_slot2": (10.0, 16.0),
                    "W_slot3": (20.0, 26.0), "Ymax": (11.0, 17.0)}, n=25, seed=2027),
}
# 獨立測試集：在模型定版後才取點，範圍 = 報告主張「模型在設計區夠準」的那一區：
# verify_best3 [36.97, 12.502, 22.944, 13.932] 周圍約 ±1.5 mm，涵蓋重訓後前 5 名與容差角落；W_slot1 下限守在 DOE 邊界 36.1。
# 用隨機（LHS）而不是最差角落：要估的是「典型」準確度，不是最壞情況。
TEST = dict(ranges={"W_slot1": (36.1, 38.5), "W_slot2": (11.0, 14.0),
                    "W_slot3": (21.5, 24.5), "Ymax": (12.5, 15.5)}, n=8, seed=2028)
TRIES = 2000


def lhs(rng, n, d):
    u = (np.argsort(rng.random((d, n)), axis=1).T + rng.random((n, d))) / n
    return u


def maximin_lhs(n, d, seed, tries=TRIES):
    """重複產生 LHS，留下「最近兩點距離」最大的一組（點越分散越好）。"""
    rng = np.random.default_rng(seed)
    best, best_d = None, -1
    for _ in range(tries):
        u = lhs(rng, n, d)
        dd = np.min([np.linalg.norm(u[i] - u[j]) for i in range(n) for j in range(i + 1, n)])
        if dd > best_d:
            best, best_d = u, dd
    return best, best_d


def write_bas(pts, label, path, out_rel=None):
    """以 run_doe.bas 為範本，換掉參數表、組數、輸出檔名。
    label：取代範本裡的 doe_batch1（例如 doe_batch2、test_set）；out_rel：輸出檔相對專案的路徑（預設 data/<label>.txt）。"""
    with open(TEMPLATE_BAS, encoding="utf-8") as f:
        text = f.read()
    if out_rel:
        text = text.replace(r"data\doe_batch1.txt", out_rel)
    n = len(pts)
    rows = "".join(f"    SetRow P, {i}, " + ", ".join(f"{x:g}" for x in p) + "\n"
                   for i, p in enumerate(pts, 1))
    text = re.sub(r"(    SetRow P, \d+,.*\n)+", lambda _: rows, text, count=1)
    text = text.replace("doe_batch1", label)
    text = re.sub(r"Const N_RUNS As Integer = \d+", f"Const N_RUNS As Integer = {n}", text)
    text = re.sub(r"P\(1 To \d+, 1 To 4\)", f"P(1 To {n}, 1 To 4)", text)
    text = re.sub(r"\d+ 組參數", f"{n} 組參數", text)
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(text.replace("\r\n", "\n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=max(BATCHES), choices=sorted(BATCHES))
    ap.add_argument("--test", action="store_true", help="產生獨立測試集（輸出到 data_test/，不會被訓練）")
    ap.add_argument("--out-dir", default=HERE, help="輸出資料夾（預設 cst/）")
    args = ap.parse_args()
    cfg = TEST if args.test else BATCHES[args.batch]
    label = "test_set" if args.test else f"doe_batch{args.batch}"
    ranges = cfg["ranges"]

    u, min_d = maximin_lhs(cfg["n"], len(ranges), cfg["seed"])
    lo = np.array([v[0] for v in ranges.values()])
    hi = np.array([v[1] for v in ranges.values()])
    pts = np.round(lo + u * (hi - lo), 1)
    pts = pts[np.lexsort(pts.T[::-1])]

    csv_path = os.path.join(args.out_dir, f"{label}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run"] + list(ranges))
        for i, p in enumerate(pts, 1):
            w.writerow([i] + [f"{x:g}" for x in p])
    print(f"{'獨立測試集' if args.test else f'第 {args.batch} 批'}：{len(pts)} 組，範圍 {ranges}")
    print(f"最小點距（正規化）= {min_d:.3f}，已存 {csv_path}")

    if args.test:
        bas_path = os.path.join(args.out_dir, "run_test_set.bas")
        write_bas(pts, label, bas_path, r"data_test\test_set.txt")
        os.makedirs(os.path.join(os.path.dirname(HERE), "data_test"), exist_ok=True)  # CST 不會自己建資料夾
        print(f"CST macro 已存 {bas_path}（輸出到 data_test/test_set.txt，不會被訓練）")
    elif args.batch != 1:  # 第 1 批的 macro 就是範本本身，不覆寫
        bas_path = os.path.join(args.out_dir, f"run_doe_batch{args.batch}.bas")
        write_bas(pts, label, bas_path)
        print(f"CST macro 已存 {bas_path}（輸出到 data/doe_batch{args.batch}.txt）")
    for i, p in enumerate(pts, 1):
        print(i, *[f"{x:g}" for x in p])


if __name__ == "__main__":
    main()
