"""
入門範例：最小的 PyTorch 代理模型（手打 5 筆假資料，2 個尺寸參數 -> S11）
[整合說明] 保留作為學習/教學用；實際 CST 資料請用 antenna_ml.py。已移除除錯用的 print("123")。
"""
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 輸入數據
# 切槽長度 (mm), 寬度 (mm)
X_train = torch.tensor([
    [10.0, 2.0],
    [12.0, 2.5],
    [14.0, 3.0],
    [11.0, 2.2],
    [13.0, 2.8]
], dtype=torch.float32)

# 對應的目標輸出db
y_train = torch.tensor([
    [-10.5],
    [-14.2],
    [-11.0],
    [-12.8],
    [-15.5]
], dtype=torch.float32)

# 2. 定義一個簡單的神經網路（代理模型 Surrogate Model）
class AntennaSurrogate(nn.Module):
    def __init__(self):
        super(AntennaSurrogate, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),   # 輸入 2 個參數，進入 8 個神經元
            nn.ReLU(),         # 激活函數
            nn.Linear(8, 1)    # 輸出 1 個預測的 S11 數值
        )
        
    def forward(self, x):
        return self.net(x)

# 初始化模型、損失函數與優化器
model = AntennaSurrogate()
criterion = nn.MSELoss()                    # 計算預測值與真實值的均方誤差
optimizer = optim.Adam(model.parameters(), lr=0.05) # 使用 Adam 優化器

# 3. 開始手動訓練 (跑 100 個 Epoch)
print("--- 開始訓練 AI 代理模型 ---")
for epoch in range(100):
    # 前向傳播：讓模型根據參數猜 S11
    predictions = model(X_train)
    
    # 計算誤差 (Loss)
    loss = criterion(predictions, y_train)
    
    # 反向傳播與參數更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每 20 次印出一次訓練狀況
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/100], Loss (誤差): {loss.item():.4f}")

print("--- 訓練完成！---\n")

# 4. 實際測試：讓 AI 預測一組全新的參數
# 假設我們丟入一個沒看過的參數：[長度 11.8 mm, 寬度 2.4 mm]
test_input = torch.tensor([[11.8, 2.4]], dtype=torch.float32)
predicted_s11 = model(test_input)

print(f"測試輸入參數 [11.8, 2.4]")
print(f"AI 預測的 S11 結果為: {predicted_s11.item():.2f} dB")