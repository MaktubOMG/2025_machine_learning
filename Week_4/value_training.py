import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== 1) 基本設定 =====
DATA_PATH = r"D:\NYCU 2527\Master program\11401\Machine Learning\ML_氣象資料\values.csv" #本機資料的路徑

# ===== 2) 讀資料
df = pd.read_csv(DATA_PATH)

# 僅取所需欄位並清洗
df = df.rename(columns={c: c.strip().lower() for c in df.columns})
df = df[["lon", "lat", "value"]].dropna()

# ===== 3) 切資料、建模、評估 =====
X = df[["lon", "lat"]].values # 從 df 中取出經度(lon)和緯度(lat)兩個欄位；.values: 轉成 NumPy二維陣列
y = df["value"].values # .values: 轉成 NumPy一維陣列

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # 把資料分成 80% 訓練集、20% 測試集；固定隨機種子，讓每次切分結果一致
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred) # 平均絕對誤差
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # 均方根誤差
r2 = r2_score(y_test, y_pred) # R square

print("===== Linear Regression (lon, lat -> temp) =====")
print(f"Coefficients of lon & lat : {model.coef_}")
print(f"Intercept : {model.intercept_:.4f}") # 常數
print(f"MAE : {mae:.4f}")  
print(f"RMSE: {rmse:.4f}")
print(f"R^2  : {r2:.4f}")

# ===== 4) 簡單視覺化：觀測 vs 預測 =====
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, s=12, alpha=0.7)
minv = min(y_test.min(), y_pred.min())
maxv = max(y_test.max(), y_pred.max())
plt.plot([minv, maxv], [minv, maxv], linestyle="--")
plt.xlabel("Observed Temp")
plt.ylabel("Predicted Temp")
plt.title("Observed vs Predicted (Linear Regression)")
plt.tight_layout()
plt.show()
