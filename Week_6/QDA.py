import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class QDA:
    def __init__(self, reg=1e-4):
        self.reg = reg  # 協方差矩陣正則化 (避免奇異)
        self.classes_ = None
        self.mu_ = {}
        self.Sigma_inv_ = {}
        self.logdet_ = {}  # log|sigma_k|
        self.logpi_ = {}   # log pi_k

    def fit(self, X, y):
        """估計各類別的 μ, Σ, π"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        n = len(y) # 全部樣本數

        for k in self.classes_:
            Xk = X[y == k]  # 類別0/1
            mu = Xk.mean(axis=0) # 類別樣本均值
            Sk = np.cov(Xk, rowvar=False) + self.reg * np.eye(X.shape[1]) # Sigma: covariance matrix
            Sinv = np.linalg.inv(Sk) 
            logdet = np.linalg.slogdet(Sk)[1]
            pi = len(Xk) / n # 0或1的機率

            self.mu_[k] = mu
            self.Sigma_inv_[k] = Sinv
            self.logdet_[k] = logdet
            self.logpi_[k] = np.log(pi)

        return self

    def _delta(self, X, k):
        """類別 k 的判別函數 δₖ(x)"""
        mu = self.mu_[k]
        Sinv = self.Sigma_inv_[k]
        Xm = X - mu
        quad = np.sum(Xm @ Sinv * Xm, axis=1)  # # 快速計算所有樣本的二次型 (x-μ)^T Σ^{-1} (x-μ)
        return -0.5 * self.logdet_[k] - 0.5 * quad + self.logpi_[k]

    def predict(self, X):
        """預測類別: 對所有 k 計算 δ_k(x)，取 argmax_k"""
        X = np.asarray(X, dtype=float)
        scores = np.column_stack([self._delta(X, k) for k in self.classes_])
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]


# 載入資料與切分
df = pd.read_csv("labels.csv")
X = df[["lon", "lat"]].to_numpy()
y = df["label"].to_numpy()

# 隨機分 80% 訓練、20% 測試
rng = np.random.default_rng(42)
n = len(X)
idx = rng.permutation(n)
split = int(n * 0.8)
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]


# 訓練與預測
qda = QDA(reg=1e-3)
qda.fit(X_train, y_train)
y_pred = qda.predict(X_test)


# 評估指標
tn = ((y_pred == 0) & (y_test == 0)).sum()
tp = ((y_pred == 1) & (y_test == 1)).sum()
fn = ((y_pred == 0) & (y_test == 1)).sum()
fp = ((y_pred == 1) & (y_test == 0)).sum()

accuracy = (tp + tn) / len(y_test)
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-12)

print("=== QDA 評估結果 ===")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1-score : {f1:.3f}")
print(f"Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")

# ========================
# 5. 繪製決策邊界
# ========================
x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = qda.predict(grid).reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], alpha=0.25, colors=["blue", "red"])
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=10, c="blue", label="Label 0", edgecolors='k', linewidths=0.3)
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=10, c="green",  label="Label 1", edgecolors='k', linewidths=0.3)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("QDA Decision Boundary")
plt.legend()
plt.tight_layout()
plt.show()
