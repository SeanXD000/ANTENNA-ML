"""產生多參數聯合取樣清單（拉丁超立方 LHS，挑出點與點之間最分散的一組）。
輸出 cst/doe_batch1.csv，供 CST 手動輸入或 run_doe.bas macro 使用。"""
import csv
import os

import numpy as np

RANGES = {  # 以目前最佳點 W_slot1=44, W_slot2=14, W_slot3=22, Ymax=18.5 為中心
    "W_slot1": (40.0, 45.0),
    "W_slot2": (12.0, 18.0),
    "W_slot3": (18.0, 26.0),
    "Ymax": (14.0, 18.5),
}
N, SEED, TRIES = 25, 2026, 2000


def lhs(rng, n, d):
    u = (np.argsort(rng.random((d, n)), axis=1).T + rng.random((n, d))) / n
    return u


rng = np.random.default_rng(SEED)
best, best_d = None, -1
for _ in range(TRIES):
    u = lhs(rng, N, len(RANGES))
    d = np.min([np.linalg.norm(u[i] - u[j]) for i in range(N) for j in range(i + 1, N)])
    if d > best_d:
        best, best_d = u, d
lo = np.array([v[0] for v in RANGES.values()])
hi = np.array([v[1] for v in RANGES.values()])
pts = np.round(lo + best * (hi - lo), 1)
pts = pts[np.lexsort(pts.T[::-1])]

out = os.path.join(os.path.dirname(__file__), "doe_batch1.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["run"] + list(RANGES))
    for i, p in enumerate(pts, 1):
        w.writerow([i] + [f"{x:g}" for x in p])
print(f"最小點距（正規化）= {best_d:.3f}，已存 {out}")
for i, p in enumerate(pts, 1):
    print(i, *[f"{x:g}" for x in p])
