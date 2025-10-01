import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1. 載入資料
# ----------------------------
df = pd.read_csv("labels.csv")  # labels.csv 已建立
X = torch.tensor(df[["lon", "lat"]].values, dtype=torch.float32)
y = torch.tensor(df["label"].values, dtype=torch.float32).unsqueeze(1)

# 隨機切分 train/test (80/20)
torch.manual_seed(42)
N = len(X)
perm = torch.randperm(N)
split = int(0.8 * N)
X_train, X_test = X[perm[:split]], X[perm[split:]]
y_train, y_test = y[perm[:split]], y[perm[split:]]

# ----------------------------
# 2. Logistic Regression 模型
# ----------------------------
class LogisticRegression(nn.Module): # 定義一個 PyTorch 模型類別，繼承 nn.Module
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1) # 建立一個線性層，輸入維度2、輸出維度1

    def forward(self, x):
        return self.linear(x)
    # 這裡沒有加sigmoid，因為後面會交由 Loss function 自己處理。

model = LogisticRegression()
criterion = nn.BCEWithLogitsLoss() # Loss function: BCE
optimizer = optim.Adam(model.parameters(), lr=0.001) # 使用Adam optimizer，會比傳統 SGD 更快收斂
                                                     #lr=0.001是學習率，控制每次更新參數的步伐大小

# ----------------------------
# 3. 訓練
# ----------------------------
epochs = 200 # 一個epoach: 把所有batch都看過一遍
train_losses = []
train_accs = []

for epoch in range(1, epochs + 1):
    logits = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 記錄訓練指標
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        acc = (preds == y_train).float().mean().item()
        
    train_losses.append(loss.item())
    train_accs.append(acc)

    if epoch % 50 == 0 or epoch == 1:
        print(f"[{epoch:03d}] Loss={loss.item():.4f} | Train Acc={acc:.3f}")

# 最終測試
model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs >= 0.5).float()
    test_acc = (test_preds == y_test).float().mean().item()
    print(f"\ntest accuracy: {test_acc:.3f}")

# ----------------------------
# 4. 視覺化
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 4.1 原始資料分布
axes[0].scatter(df.loc[df["label"]==0,"lon"], df.loc[df["label"]==0,"lat"], 
                c="blue", label="label=0", alpha=0.7, s=20)
axes[0].scatter(df.loc[df["label"]==1,"lon"], df.loc[df["label"]==1,"lat"], 
                c="red", label="label=1", alpha=0.7, s=20)
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title("Raw data distribution")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 4.2 訓練曲線
axes[1].plot(train_losses, 'b-', label='Training Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 在右側y軸顯示準確率
ax1_twin = axes[1].twinx()
ax1_twin.plot(train_accs, 'r--', label='Training Accuracy')
ax1_twin.set_ylabel('Accuracy')
ax1_twin.legend(loc='lower right')

# 4.3 決策邊界
w1, w2 = model.linear.weight[0].detach().numpy()
b = model.linear.bias.item()

# 決策邊界: w1*x1 + w2*x2 + b = 0
x_vals = np.linspace(df["lon"].min(), df["lon"].max(), 200)
y_vals = -(w1 * x_vals + b) / w2

axes[2].scatter(df.loc[df["label"]==0,"lon"], df.loc[df["label"]==0,"lat"], 
                c="blue", label="label=0", alpha=0.7, s=20)
axes[2].scatter(df.loc[df["label"]==1,"lon"], df.loc[df["label"]==1,"lat"], 
                c="red", label="label=1", alpha=0.7, s=20)
axes[2].plot(x_vals, y_vals, 'k--', linewidth=2, label="class_line")
axes[2].set_xlabel("Longitude")
axes[2].set_ylabel("Latitude")
axes[2].set_title("Logistic Regression")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 顯示數據統計
print(f"\nData statistics:")
print(f"Total: {len(df)}")
print(f"label=0 : {len(df[df['label']==0])} ({len(df[df['label']==0])/len(df)*100:.1f}%)")
print(f"label=1 : {len(df[df['label']==1])} ({len(df[df['label']==1])/len(df)*100:.1f}%)")