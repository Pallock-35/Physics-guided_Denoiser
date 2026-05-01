import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pywt

from utils import advdiff_fd, add_noise, compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs("results/baselines", exist_ok=True)

# config
X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 0.1
NX, NT       = 301, 301
K            = 0.0025
C_TRUE       = 1.0
SIGMA0, X0   = 0.06, 0.25

SIGMA_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25]
TEST_SEEDS   = [111, 222, 333]

# domain 
x_lin = np.linspace(X_MIN, X_MAX, NX)
t_lin = np.linspace(T_MIN, T_MAX, NT)
X, T  = np.meshgrid(x_lin, t_lin)
x_all = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
t_all = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)
N_ALL = x_all.shape[0]

# true solution
u0_ic = np.exp(-((x_lin - X0)**2) / SIGMA0**2)
u_true_2d   = advdiff_fd(x_lin, t_lin, u0_ic, K, C_TRUE)
u_true_flat = u_true_2d.reshape(-1).astype(np.float32)
amp = float(np.max(u_true_flat))

# Baseline 1: Gaussian filter 
def gaussian_filter_2d(u_2d, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d
    out = np.zeros_like(u_2d)
    for ti in range(u_2d.shape[0]):
        out[ti] = gaussian_filter1d(u_2d[ti], sigma=sigma)
    return out

# Baseline 2: Wavelet denoising
def wavelet_denoise_2d(u_2d, wavelet="haar", level=1):
    out = np.zeros_like(u_2d)
    for ti in range(u_2d.shape[0]):
        s = u_2d[ti].astype(np.float64)
        coeffs = pywt.wavedec(s, wavelet, level=level)
        thr = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(s)))
        coeffs[1:] = [pywt.threshold(c, thr, "soft") for c in coeffs[1:]]
        out[ti] = pywt.waverec(coeffs, wavelet)[:len(s)]
    return out

# Baseline 3: PINN-noisy
class PINNNoisy(nn.Module):
    def __init__(self, width=128, depth=6):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def train_pinn_noisy(sig_frac, seed, epochs=4000):
    torch.manual_seed(seed)
    u_obs_flat = add_noise(u_true_flat, sig_frac, amp, seed=seed)
    u_obs_all  = torch.tensor(u_obs_flat.reshape(-1, 1), dtype=torch.float32, device=device)

    # train/val split
    perm   = torch.randperm(N_ALL, device=device)
    n_tr   = int(0.8 * N_ALL)
    idx_tr = perm[:n_tr]; idx_va = perm[n_tr:]

    model = PINNNoisy().to(device)
    opt   = optim.AdamW(model.parameters(), lr=2e-3)
    mse   = nn.MSELoss()
    BS    = 4096

    def diff_residual(xb, tb):
        xb = xb.clone().requires_grad_(True)
        tb = tb.clone().requires_grad_(True)
        u    = model(xb, tb)
        u_t  = torch.autograd.grad(u,   tb, torch.ones_like(u),   create_graph=True)[0]
        u_x  = torch.autograd.grad(u,   xb, torch.ones_like(u),   create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, xb, torch.ones_like(u_x), create_graph=True)[0]
        return u_t - K * u_xx

    best_val, best_state, bad = float("inf"), None, 0

    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        idx  = torch.randint(0, n_tr, (BS,), device=device)
        xb, tb, yb = x_all[idx_tr][idx], t_all[idx_tr][idx], u_obs_all[idx_tr][idx]
        l_data = mse(model(xb, tb), yb)
        l_pde  = torch.mean(diff_residual(xb, tb)**2)
        loss   = l_data + l_pde
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if ep % 200 == 0:
            model.eval()
            with torch.no_grad():
                idxv = torch.randint(0, N_ALL - n_tr, (BS,), device=device)
                val = mse(model(x_all[idx_va][idxv], t_all[idx_va][idxv]),
                          u_obs_all[idx_va][idxv]).item()
            if val < best_val - 1e-5:
                best_val   = val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 15: break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        return model(x_all, t_all).cpu().numpy().reshape(-1)

rows = []

for sig in SIGMA_LEVELS:
    print(f"\nsigma={sig:.2f}")
    for seed in TEST_SEEDS:
        print(f"  seed {seed}", end=" ", flush=True)
        u_obs_flat = add_noise(u_true_flat, sig, amp, seed=seed)
        u_obs_2d   = u_obs_flat.reshape(NT, NX)

        # Gaussian
        xg = gaussian_filter_2d(u_obs_2d, sigma=1.0).reshape(-1)
        # Wavelet
        xw_2d = wavelet_denoise_2d(u_obs_2d, wavelet="haar", level=1)
        xw    = xw_2d.reshape(-1) if xw_2d is not None else None
        # PINN-noisy
        xp = train_pinn_noisy(sig, seed)
        print("done")

        for name, pred in [("Gaussian", xg), ("Wavelet", xw), ("PINN_noisy", xp)]:
            if pred is None: continue
            m = compute_metrics(u_true_flat, pred)
            rows.append({"sigma": sig, "seed": seed, "method": name, **m})

# summary
for sig in SIGMA_LEVELS:
    for method in ("Gaussian", "Wavelet", "PINN_noisy"):
        sub  = [r for r in rows if r["sigma"] == sig and r["method"] == method]
        if not sub: continue
        for key in ("RMSE", "R2", "SNR_dB"):
            vals = np.array([r[key] for r in sub])
            print(f"  sigma={sig:.2f}  {method:12s}  {key}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")

# save
with open("results/baselines/metrics.csv", "w") as f:
    f.write("sigma,seed,method,RMSE,R2,SNR_dB\n")
    for r in rows:
        f.write(f"{r['sigma']},{r['seed']},{r['method']},{r['RMSE']:.6f},{r['R2']:.6f},{r['SNR_dB']:.4f}\n")

print("\nDone.")
