"""
Stage 1: QDA classifier — 判斷每個格點是否有有效值 (label=1)
Stage 2: Linear Regression — 對有值的點預測溫度，使用 MSE loss 評估誤差
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# QDA from scratch
# ============================================================
class QDA:
    """二次判別分析 (Quadratic Discriminant Analysis)"""
    def __init__(self, reg: float = 1e-3):
        self.reg = reg
        self.classes_ = None
        self.mu_, self.Sigma_inv_, self.logdet_, self.logpi_ = {}, {}, {}, {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        n = len(y)
        d = X.shape[1]

        for k in self.classes_:
            Xk = X[y == k]
            mu = Xk.mean(axis=0)
            Sk = np.cov(Xk, rowvar=False, bias=False)
            Sk = Sk + self.reg * np.eye(d)
            Sinv = np.linalg.inv(Sk)
            logdet = np.linalg.slogdet(Sk)[1]
            pi = len(Xk) / n

            self.mu_[k] = mu
            self.Sigma_inv_[k] = Sinv
            self.logdet_[k] = logdet
            self.logpi_[k] = np.log(max(pi, 1e-12))

        return self

    def _delta(self, X, k):
        mu = self.mu_[k]
        Sinv = self.Sigma_inv_[k]
        Xm = X - mu
        quad = np.sum((Xm @ Sinv) * Xm, axis=1)
        return -0.5 * self.logdet_[k] - 0.5 * quad + self.logpi_[k]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.column_stack([self._delta(X, k) for k in self.classes_])
        return self.classes_[np.argmax(scores, axis=1)]

# ============================================================
# Linear Regression (OLS)
# ============================================================
class LinearRegression:
    """簡單線性回歸：y = Xβ，以最小平方法 (OLS) 求解"""
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        Xd = np.column_stack([np.ones(len(X)), X])   # 加上截距項
        beta = np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ y  # OLS 解
        self.beta_ = beta
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xd = np.column_stack([np.ones(len(X)), X])
        return (Xd @ self.beta_).ravel()

# ============================================================
# Load Data
# ============================================================
labels_path = Path("labels.csv")
values_path = Path("values.csv")

df_lab = pd.read_csv(labels_path)  # lon, lat, label
df_val = pd.read_csv(values_path)  # lon, lat, value

# 基本欄位檢查
assert {"lon","lat","label"}.issubset(df_lab.columns)
assert {"lon","lat","value"}.issubset(df_val.columns)

# X, y for classification
X_all = df_lab[["lon","lat"]].to_numpy(dtype=float)
y_all = df_lab["label"].to_numpy(dtype=int)

# X, y for regression (only valid points)
X_reg = df_val[["lon","lat"]].to_numpy(dtype=float)
y_reg = df_val["value"].to_numpy(dtype=float)

# ============================================================
# Train Models
# ============================================================
qda = QDA(reg=1e-3).fit(X_all, y_all)
linreg = LinearRegression().fit(X_reg, y_reg)
beta0, beta1, beta2 = linreg.beta_.ravel()
print("\n=== Linear Regression Parameters ===")
print(f"Equation: y = {beta0:.3f} + {beta1:.3f}·lon + {beta2:.3f}·lat")

# ============================================================
# Prediction
# ============================================================
pred_label = qda.predict(X_all)
pred_value = np.full(len(X_all), -999.0)
mask_valid = pred_label == 1
pred_value[mask_valid] = linreg.predict(X_all[mask_valid])

# ============================================================
# Merge & Evaluate
# ============================================================
out = df_lab.rename(columns={"label":"true_label"}).copy()
out["pred_label"] = pred_label
out["pred_value"] = pred_value

# Merge with true value (for MSE evaluation)
out = out.merge(df_val, on=["lon","lat"], how="left", suffixes=("",""))

# 分類指標
tn = int(((out["pred_label"]==0) & (out["true_label"]==0)).sum())
tp = int(((out["pred_label"]==1) & (out["true_label"]==1)).sum())
fp = int(((out["pred_label"]==1) & (out["true_label"]==0)).sum())
fn = int(((out["pred_label"]==0) & (out["true_label"]==1)).sum())
acc = (tp + tn) / len(out)
prec = tp / max(tp + fp, 1)
rec  = tp / max(tp + fn, 1)
f1   = 2 * prec * rec / max(prec + rec, 1e-12)
print("\n=== Classification (QDA) ===")
print(f"Accuracy : {acc:.3f} | Precision : {prec:.3f} | Recall : {rec:.3f} | F1 : {f1:.3f}")
print(f"Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")

# 回歸 MSE（僅針對 pred_label=1 且有真值者）
mask_eval = (out["pred_label"]==1) & (~out["value"].isna())
if mask_eval.any():
    y_true = out.loc[mask_eval, "value"].to_numpy(dtype=float)
    y_pred = out.loc[mask_eval, "pred_value"].to_numpy(dtype=float)
    mse  = np.mean((y_pred - y_true)**2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(y_pred - y_true))
    print("\n=== Regression (Linear) ===")
    print(f"MSE={mse:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | N={len(y_true)}")
else:
    print("\n[Note] No valid points for regression evaluation.")

# ============================================================
# Visualization
# ============================================================
# QDA 橢圓
x_min, x_max = X_all[:,0].min()-0.05, X_all[:,0].max()+0.05
y_min, y_max = X_all[:,1].min()-0.05, X_all[:,1].max()+0.05
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 600),
    np.linspace(y_min, y_max, 600)
)
grid = np.c_[xx.ravel(), yy.ravel()]
zz = qda.predict(grid).reshape(xx.shape)
plt.figure(figsize=(7,6))
plt.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.2, linestyles="--")
plt.scatter(X_all[y_all==0,0], X_all[y_all==0,1],
            s=15, c="#1f78b4", edgecolors="k", linewidths=0.3, label="Label = 0")
plt.scatter(X_all[y_all==1,0], X_all[y_all==1,1],
            s=15, c="#098943", edgecolors="k", linewidths=0.3, label="Label = 1")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("QDA Decision Boundary + Linear Regression Region")
plt.axis("equal")
plt.legend(frameon=True, loc="upper right")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# True vs. prediction
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, s=20, alpha=0.7, c="dodgerblue", edgecolors="k", linewidths=0.3)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="Ideal (y=x)")
plt.xlabel("True Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title(f"True vs Predicted (RMSE={rmse:.2f} °C)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# 僅分析有真值 (value) 且被預測為 label=1 的點
mask_eval = (out["pred_label"] == 1) & (~out["value"].isna())
y_true = out.loc[mask_eval, "value"].to_numpy(dtype=float)
y_pred = out.loc[mask_eval, "pred_value"].to_numpy(dtype=float)
errors = y_pred - y_true  # 誤差 (預測 - 實際)

# 空間誤差地圖 (Spatial Error Map) ---
plt.figure(figsize=(8,6))
plt.scatter(out.loc[mask_eval, "lon"], out.loc[mask_eval, "lat"],
            c=errors, cmap="coolwarm", s=30, edgecolors="k", linewidths=0.3)
plt.colorbar(label="Prediction Error (°C)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Distribution of Temperature Prediction Errors")
plt.axis("equal")
plt.tight_layout()
plt.show()


