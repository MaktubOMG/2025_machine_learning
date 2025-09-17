# nn; Neural Network(神經網路)
import math
import numpy as np
import torch #機器學習框架
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# torch.Tensor: PyTorch的核心數據結構，是一個多維數組，非常類似於 NumPy 的 ndarray。它主要用來儲存和處理神經網絡中的數據。
# -----------------------------
# 1) Target function & derivative
# -----------------------------
def runge(x: torch.Tensor) -> torch.Tensor: 
    """Runge function f(x) = 1/(1+25 x^2). x shape: (N,1)"""
    return 1.0 / (1.0 + 25.0 * x**2)

def runge_dx(x: torch.Tensor) -> torch.Tensor:
    """Analytic derivative f'(x) = -50 x / (1+25 x^2)^2"""
    return -50.0 * x / (1.0 + 25.0 * x**2)**2

# -----------------------------
# 2) Chebyshev-like sampler & weight
# -----------------------------
# 使樣本能更密集分布在端點，以減少Runge function在端點震盪所造成的近似誤差
def sample_chebyshev(n: int, device="cpu") -> torch.Tensor:
    """u~U(0,1), x=cos(pi*u): denser near ±1; returns (n,1) in [-1,1]."""
    u = torch.rand(n, 1, device=device)
    x = torch.cos(math.pi * u)
    return x

# 決定誤差加權（weighting）**的方式，用來強化端點的影響力。
def chebyshev_weight(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    w(x) = 1 / sqrt(1 - x^2 + eps), normalized to mean 1.
    Larger near endpoints; used to upweight errors there.
    """
    w = 1.0 / torch.sqrt(torch.clamp(1.0 - x**2, min=0.0) + eps)
    return w / w.mean()

# -----------------------------
# 3) Model: 2-hidden-layer tanh MLP
# -----------------------------
# 使用兩層hidden layer，每層神經元數量為64(深度為2，寬度為64):(1-64-64-1)
class MLP2(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.l1 = nn.Linear(1, width)
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, 1)
        self.reset_parameters_tanh()
    # 初始化神經網路層參數，gain: 計算tanh activation function 對應的增益值，用來修正方差
    # 以 Xavier/Glorot-uniform 方法初始化權重，確保前向與反向訊號的方差平衡。
    def reset_parameters_tanh(self):
        """Xavier/Glorot-uniform with tanh gain for each Linear layer; bias=0."""
        gain = nn.init.calculate_gain('tanh')  # ~ 5/3
        for layer in (self.l1, self.l2, self.l3):
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias) # 將bias設為0，避免初始偏移
    # 向前傳遞，return 估計值 y
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(self.l1(x))
        h2 = torch.tanh(self.l2(h1))
        y  = self.l3(h2)     # linear output (no activation)
        return y

# -----------------------------
# 4) Training utilities
# -----------------------------
@torch.no_grad()
# 計算最大絕對誤差、RMSE(均方差) 
def eval_metrics(model: nn.Module, xs: torch.Tensor) -> dict:
    """Compute RMSE and L∞ on a dense grid xs."""
    y_true = runge(xs)
    y_pred = model(xs)
    err = (y_pred - y_true).abs()
    rmse = (err.pow(2).mean()).sqrt().item()
    linf = err.max().item()
    return {"RMSE": rmse, "L_inf": linf}

def train(
    steps: int = 3000,   # 訓練總步數:3000
    batch_size: int = 512, #每步使用之樣本數。較大批量可穩定梯度、提升端點加權的統計性
    width: int = 64,
    alpha: float = 0.7,  # alpha,beta: 總loss中，未加權MSE與端點加權MSE之組合係數(α>β)
    beta: float = 0.3,
    use_sobolev: bool = False,
    lambda_d: float = 0.1,
    lr: float = 1e-3, # AdamW 之學習率(關鍵超參數)(優化器是可以指定的，SGD收斂可能較慢，因此選用結合各優勢之AdamW)
    weight_decay: float = 1e-6,
    device: str = "cpu",
    seed: int = 0 # 隨機種子，用以固定取樣與初始化，使實驗可重現
):
    torch.manual_seed(seed)
    model = MLP2(width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_hist = []
    grid = torch.linspace(-1.0, 1.0, 2000, device=device).unsqueeze(1)  # for periodic eval

    for t in range(1, steps+1):
        # ---- sample a mini-batch (Chebyshev-like) ----
        x = sample_chebyshev(batch_size, device=device).requires_grad_(use_sobolev)
        y = runge(x)

        # ---- forward ----
        yhat = model(x)
        e = yhat - y # 誤差

        # ---- loss: alpha * MSE + beta * weighted MSE ----
        if beta > 0.0:
            w = chebyshev_weight(x)
            mse = (e.pow(2)).mean()
            wmse = ((w * e).pow(2)).mean()
            loss = alpha * mse + beta * wmse
        else:
            loss = (e.pow(2)).mean()

        # ---- optional Sobolev (derivative) term ----
        if use_sobolev:
            grad = torch.autograd.grad(yhat.sum(), x, create_graph=False)[0]
            sob = (grad - runge_dx(x)).pow(2).mean()
            loss = loss + lambda_d * sob

        # ---- backward & step ----
        opt.zero_grad(set_to_none=True) # 將前一次迭代所累積的梯度清除
        loss.backward() # 自動計算損失函數對模型參數的梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 對所有參數的梯度進行梯度裁切(整體梯度向量之L2 norm不超過1)
        opt.step()

        loss_hist.append(loss.item())

        # ---- light logging ----
        if t % 500 == 0 or t == 1 or t == steps: #每500步、第一步、以及最後一步，執行評估並印出來
            with torch.no_grad():
                metrics = eval_metrics(model, grid)
            print(f"step {t:5d} | loss={loss.item():.4e} | RMSE={metrics['RMSE']:.4e} | L∞={metrics['L_inf']:.4e}")

    # ---- final plots ----
    with torch.no_grad():
        xs = torch.linspace(-1.0, 1.0, 2000, device=device).unsqueeze(1)
        y_true = runge(xs)
        y_pred = model(xs)
        err = (y_pred - y_true).abs()

    xs_np = xs.squeeze(1).cpu().numpy()
    y_np  = y_true.squeeze(1).cpu().numpy()
    yhat_np = y_pred.squeeze(1).cpu().numpy()
    err_np  = err.squeeze(1).cpu().numpy()

    # Plot 1: f vs f_hat 
    plt.figure(figsize=(7,4))
    plt.plot(xs_np, y_np, label="Runge f(x)")
    plt.plot(xs_np, yhat_np, label="NN approximation")
    plt.title("Runge vs NN (tanh, 2HL, Chebyshev sampling)")
    plt.xlabel("x"); plt.ylabel("y"); plt.grid(True, alpha=0.3); plt.legend()
    plt.show()

    # Plot 2: absolute error # 誤差絕對值圖表
    plt.figure(figsize=(7,4))
    plt.plot(xs_np, err_np, label="|f̂ - f|")
    plt.title("Absolute error across [-1,1]")
    plt.xlabel("x"); plt.ylabel("abs error"); plt.grid(True, alpha=0.3); plt.legend()
    plt.show()

    # Plot 3: training loss #訓練損失曲線圖
    plt.figure(figsize=(7,4))
    plt.plot(np.arange(1, len(loss_hist)+1), loss_hist)
    plt.title("Training loss")
    plt.xlabel("step"); plt.ylabel("loss"); plt.grid(True, alpha=0.3)
    plt.show()

    # Final metrics
    metrics = eval_metrics(model, torch.linspace(-1,1,5000, device=device).unsqueeze(1))
    print("\n=== Final metrics (dense grid) ===")
    print(f"RMSE  = {metrics['RMSE']:.6e}")
    print(f"L_inf = {metrics['L_inf']:.6e}")

    return model

if __name__ == "__main__": #主程式
    _ = train(
        steps=3000,   # 如果要更精確可以調大
        batch_size=512,
        width=64,
        alpha=0.7, beta=0.3, # beta=0代表移除端點權重
        use_sobolev=False,   
        lambda_d=0.1,
        lr=1e-3, weight_decay=1e-6,
        device="cpu",        
        seed=0
    )