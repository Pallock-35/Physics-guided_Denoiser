import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import advdiff_fd, add_noise, compute_metrics, ebm_nll, get_rmax

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs("results/stage23", exist_ok=True)

# config
X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 0.1
NX, NT       = 301, 301
K            = 0.0025   # diffusion coefficient (known)
C_TRUE       = 1.0      # advection speed (unknown to surrogate)
SIGMA0, X0   = 0.06, 0.25

TRAIN_NOISE  = 0.10
SEED_DATA    = 1000

SIGMA_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25]
TEST_SEEDS   = [111, 222, 333]

LAM1, LAM2   = 1.0, 0.05    # data fidelity and EBM reg weights
LAM_OVER     = 0.10         # overflow penalty weight

# domain 
x_lin = np.linspace(X_MIN, X_MAX, NX)
t_lin = np.linspace(T_MIN, T_MAX, NT)
X, T  = np.meshgrid(x_lin, t_lin)
x_all = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
t_all = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)
N_ALL = x_all.shape[0]

# true solution (advection + diffusion)
u0_ic = np.exp(-((x_lin - X0)**2) / SIGMA0**2)
u_true_fd = advdiff_fd(x_lin, t_lin, u0_ic, K, C_TRUE)
u_true_flat = u_true_fd.reshape(-1).astype(np.float32)
amp = float(np.max(u_true_flat))
print(f"Domain: {NX}x{NT} points  |  amp={amp:.4f}")

# load Stage-I surrogate (frozen)
class PINN_Diff(nn.Module):
    def __init__(self, width=96, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

ckpt = torch.load("results/stage1/surrogate.pth", map_location=device)
surrogate = PINN_Diff().to(device)
surrogate.load_state_dict(ckpt["state"])
surrogate.eval()
for p in surrogate.parameters():
    p.requires_grad_(False)

with torch.no_grad():
    f_all = surrogate(x_all, t_all)

# training observations (fixed seed)
u_obs_flat = add_noise(u_true_flat, TRAIN_NOISE, amp, seed=SEED_DATA)
u_obs_all  = torch.tensor(u_obs_flat.reshape(-1, 1), dtype=torch.float32, device=device)

# train/val split 
torch.manual_seed(SEED_DATA)
perm    = torch.randperm(N_ALL, device=device)
n_tr    = int(0.8 * N_ALL)
idx_tr, idx_va = perm[:n_tr], perm[n_tr:]

x_tr, t_tr = x_all[idx_tr], t_all[idx_tr]
x_va, t_va = x_all[idx_va], t_all[idx_va]
y_tr, y_va = u_obs_all[idx_tr], u_obs_all[idx_va]
f_tr, f_va = f_all[idx_tr],     f_all[idx_va]

# model definitions 
class EBM_xt(nn.Module):
    def __init__(self, width=128, depth=4):
        super().__init__()
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x, t, r):
        return self.net(torch.cat([x, t, r], dim=1))

class DenoiserNet(nn.Module):
    def __init__(self, width=128, depth=5):
        super().__init__()
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x, t, y):
        return self.net(torch.cat([x, t, y], dim=1))

ebm      = EBM_xt().to(device)
denoiser = DenoiserNet().to(device)

# Stage II: train EBM
with torch.no_grad():
    r_tr = y_tr - f_tr
    r_va = y_va - f_va

ebm_opt = optim.AdamW(ebm.parameters(), lr=2e-3)
ebm_sch = optim.lr_scheduler.CosineAnnealingLR(ebm_opt, T_max=6000, eta_min=1e-5)

best_val, best_state = float("inf"), None
BS = 512

for ep in range(1, 6001):
    ebm.train()
    ebm_opt.zero_grad()
    idx  = torch.randint(0, n_tr, (BS,), device=device)
    xb, tb, rb = x_tr[idx], t_tr[idx], r_tr[idx]
    rmax = get_rmax(rb)
    loss = ebm_nll(xb, tb, rb, ebm, -rmax, rmax)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ebm.parameters(), 5.0)
    ebm_opt.step(); ebm_sch.step()

    if ep % 500 == 0:
        ebm.eval()
        with torch.no_grad():
            idxv  = torch.randint(0, len(idx_va), (BS,), device=device)
            rmax_v = get_rmax(r_va[idxv])
            val   = ebm_nll(x_va[idxv], t_va[idxv], r_va[idxv], ebm, -rmax_v, rmax_v).item()
        if val < best_val:
            best_val   = val
            best_state = {k: v.cpu().clone() for k, v in ebm.state_dict().items()}
        print(f"  ep {ep}  train={loss.item():.3e}  val={val:.3e}")

ebm.load_state_dict(best_state)
torch.save(ebm.state_dict(), "results/stage23/ebm.pth")
for p in ebm.parameters():
    p.requires_grad_(False)
ebm.eval()

# Stage III: train denoiser
den_opt = optim.AdamW(denoiser.parameters(), lr=2e-3)
den_sch = optim.lr_scheduler.CosineAnnealingLR(den_opt, T_max=9000, eta_min=1e-5)
mse_fn  = nn.MSELoss()

with torch.no_grad():
    rmax_global = get_rmax(r_tr)

best_val, best_state, patience, bad = float("inf"), None, 500, 0
BS = 1024

for ep in range(1, 9001):
    denoiser.train()
    den_opt.zero_grad()

    idx      = torch.randint(0, n_tr, (BS,), device=device)
    xb, tb   = x_tr[idx], t_tr[idx]
    yb, fb   = y_tr[idx], f_tr[idx]
    u0       = denoiser(xb, tb, yb)
    l_data   = mse_fn(u0, yb)
    r0       = u0 - fb
    l_ebm    = ebm_nll(xb, tb, r0, ebm, -rmax_global, rmax_global)
    l_over   = (torch.relu(r0.abs() - rmax_global)**2).mean()
    loss     = LAM1 * l_data + LAM2 * l_ebm + LAM_OVER * l_over
    loss.backward()
    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 5.0)
    den_opt.step(); den_sch.step()

    denoiser.eval()
    with torch.no_grad():
        idxv    = torch.randint(0, len(idx_va), (BS,), device=device)
        xv, tv  = x_va[idxv], t_va[idxv]
        u0v     = denoiser(xv, tv, y_va[idxv])
        l_dv    = mse_fn(u0v, y_va[idxv])
        r0v     = u0v - f_va[idxv]
        l_ev    = ebm_nll(xv, tv, r0v, ebm, -rmax_global, rmax_global)
        val     = (LAM1 * l_dv + LAM2 * l_ev).item()

    if val < best_val - 1e-6:
        best_val   = val
        best_state = {k: v.cpu().clone() for k, v in denoiser.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= patience:
            print(f"  early stop at ep {ep}"); break

    if ep % 1000 == 0:
        print(f"  ep {ep}  train={loss.item():.3e}  val={val:.3e}")

denoiser.load_state_dict(best_state)
torch.save(denoiser.state_dict(), "results/stage23/denoiser.pth")

# evaluation
denoiser.eval()
rows = []

for sig in SIGMA_LEVELS:
    for seed in TEST_SEEDS:
        u_test_flat = add_noise(u_true_flat, sig, amp, seed=seed)
        u_test_all  = torch.tensor(u_test_flat.reshape(-1, 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            u0_all = denoiser(x_all, t_all, u_test_all).cpu().numpy().reshape(-1)
        m = compute_metrics(u_true_flat, u0_all)
        rows.append({"sigma": sig, "seed": seed, **m})

for sig in SIGMA_LEVELS:
    sub = [r for r in rows if r["sigma"] == sig]
    for key in ("RMSE", "R2", "SNR_dB"):
        vals = np.array([r[key] for r in sub])
        print(f"  sigma={sig:.2f}  {key}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")

# save csv
with open("results/stage23/metrics.csv", "w") as f:
    f.write("sigma,seed,RMSE,R2,SNR_dB\n")
    for r in rows:
        f.write(f"{r['sigma']},{r['seed']},{r['RMSE']:.6f},{r['R2']:.6f},{r['SNR_dB']:.4f}\n")

# representative plot at sigma=0.25
sig, seed = 0.25, TEST_SEEDS[0]
u_test = add_noise(u_true_flat, sig, amp, seed=seed)
u_test_all = torch.tensor(u_test.reshape(-1, 1), dtype=torch.float32, device=device)
with torch.no_grad():
    u0_plot = denoiser(x_all, t_all, u_test_all).cpu().numpy().reshape(NT, NX)

t_idx = NT // 2   # mid-time slice
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(x_lin, u_true_fd[t_idx],  label="true")
ax.plot(x_lin, u_test.reshape(NT, NX)[t_idx], alpha=0.4, label="noisy")
ax.plot(x_lin, u0_plot[t_idx],    label="denoised", lw=2)
ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/stage23/denoised_25pct_tslice.png", dpi=200)
plt.close(fig)
print("Done.")
