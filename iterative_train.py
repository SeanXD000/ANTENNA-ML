"""
疊代微調 (Iterative Fine-tuning)
------------------------------------------------
[整合說明] 本檔合併了舊的 iterative_train.py（手打 2 筆資料、2->8->1 小網路）與
          iterative_train_R1.py（讀 ./data 的 txt，但假設欄位是 length/width，與 CST 實際匯出格式不符而無法執行）。
          現在直接沿用 antenna_ml.py 的 CST 解析器、模型與損失，對同一個 antenna_surrogate.pth 做「繼承式」更新。

流程：
 1. 讀取 antenna_ml.py 存下的集成模型 antenna_surrogate.pth（沒有就從頭訓練一份）
 2. 讀取 ./data 內所有 CST 匯出檔（把新補跑的模擬丟進 data/ 即可）
 3. 用較小學習率微調每個集成成員，印出微調前後的 RMSE
 4. 覆寫存回 antenna_surrogate.pth，供下次疊代或 antenna_ml.py 使用

用法：
  python iterative_train.py                    # 預設微調 300 epochs
  python iterative_train.py --epochs 500 --lr 1e-4 --band WIFI7
注意：正規化範圍沿用第一次訓練時的設定；若新資料的參數範圍明顯超出，建議改跑 antenna_ml.py 重新訓練。
"""
import argparse
import os

import numpy as np
import torch

from antenna_ml import (DATA_DIR, DEVICE, MODEL_PATH, WIFI_BANDS, AntennaSurrogate, Normalizer,
                        compute_sample_weights, load_all_runs, runs_to_arrays, train_ensemble)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    norm = Normalizer(**ckpt["norm"])
    models = []
    for sd in ckpt["state_dicts"]:
        m = AntennaSurrogate(len(ckpt["design_params"])).to(DEVICE)
        m.load_state_dict(sd)
        models.append(m)
    return models, norm, ckpt["design_params"]


def to_tensors(runs, design_params, norm, bands):
    P, F, Y = runs_to_arrays(runs, design_params)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    return (t(norm.p(P)), t(norm.f(F)).unsqueeze(1),
            t((Y - norm.y_mean) / norm.y_std).unsqueeze(1),
            t(compute_sample_weights(F, Y, bands)).unsqueeze(1))


@torch.no_grad()
def rmse_db(model, p, f, y, norm):
    model.eval()
    return float(torch.sqrt(torch.mean((model(p, f) - y) ** 2))) * norm.y_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="略過的檔名樣式（預設同 antenna_ml.py：W4*）")
    ap.add_argument("--band", default="6E", choices=list(WIFI_BANDS))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-5, help="微調學習率；太大 (如 2e-4) 會先把已收斂的模型打亂")
    args = ap.parse_args()
    bands = WIFI_BANDS[args.band]

    runs, design_params, fixed = load_all_runs(args.data, args.exclude)
    print(f"成功載入 {len(runs)} 組模擬，設計變數：{design_params}")

    if not os.path.exists(MODEL_PATH):
        print(f"--- 未發現 {MODEL_PATH}，從頭訓練新模型 ---")
        models, norm = train_ensemble(runs, design_params, bands)
    else:
        models, norm, saved_params = load_checkpoint(MODEL_PATH)
        if saved_params != design_params:
            raise SystemExit(f"設計變數不一致：模型是 {saved_params}，資料是 {design_params}。"
                             f"請改跑 antenna_ml.py 重新訓練。")
        print(f"--- 成功載入舊模型 {MODEL_PATH}（{len(models)} 個集成成員），開始疊代微調 ---")

        p, f, y, w = to_tensors(runs, design_params, norm, bands)
        n, bs = len(y), min(4096, len(y))
        for i, model in enumerate(models, 1):
            before = rmse_db(model, p, f, y, norm)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            for _ in range(args.epochs):
                perm = torch.randperm(n, device=DEVICE)
                for j in range(0, n, bs):
                    idx = perm[j:j + bs]
                    loss = torch.mean(w[idx] * (model(p[idx], f[idx]) - y[idx]) ** 2)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            print(f"  [模型 {i}/{len(models)}] RMSE {before:.3f} dB -> {rmse_db(model, p, f, y, norm):.3f} dB")

    torch.save({"state_dicts": [m.state_dict() for m in models], "norm": norm.state(),
                "design_params": design_params, "fixed_params": fixed}, MODEL_PATH)
    print(f"--- 模型已更新並存檔至：{MODEL_PATH} ---")


if __name__ == "__main__":
    main()
