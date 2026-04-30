import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models import Surrogate, EBM, Denoiser
from utils  import damped_sho, add_noise, split_indices, compute_metrics, ebm_nll, get_rmax

torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('results/stage23', exist_ok=True)

# config
T0, T1, N = 0.0, 15.0, 5000
TRAIN_NOISE = 0.10
SEED_DATA   = 999

SIGMA_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25]
TEST_SEEDS   = [111, 222, 333]

LAM1, LAM2 = 1.0, 0.05

# data
t_all  = torch.linspace(T0, T1, N, device=device).view(-1, 1)
x_true = damped_sho(t_all).to(device)
std_ref = float(x_true.std())

with torch.no_grad():
    surrogate = Surrogate().to(device)
    ckpt = torch.load('results/stage1/surrogate.pth', map_location=device)
    surrogate.load_state_dict(ckpt['state'])
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)
    f_all = surrogate(t_all)

x_obs = add_noise(x_true, TRAIN_NOISE, std_ref, seed=SEED_DATA)

idx_tr, idx_va, idx_te = split_indices(N, device=device)
t_tr, t_va = t_all[idx_tr], t_all[idx_va]
y_tr, y_va = x_obs[idx_tr], x_obs[idx_va]
f_tr, f_va = f_all[idx_tr], f_all[idx_va]

# Stage II: train EBM
ebm     = EBM().to(device)
ebm_opt = optim.AdamW(ebm.parameters(), lr=2e-3)
ebm_sch = optim.lr_scheduler.CosineAnnealingLR(ebm_opt, T_max=9000, eta_min=1e-5)

with torch.no_grad():
    r_tr = y_tr - f_tr
    r_va = y_va - f_va

best_val, best_state = float('inf'), None

for ep in range(1, 9001):
    ebm.train()
    ebm_opt.zero_grad()
    rmax = get_rmax(r_tr)
    loss = ebm_nll(t_tr, r_tr, ebm, -rmax, rmax)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ebm.parameters(), 5.0)
    ebm_opt.step(); ebm_sch.step()

    if ep % 500 == 0:
        ebm.eval()
        with torch.no_grad():
            rmax_v = get_rmax(r_va)
            val = ebm_nll(t_va, r_va, ebm, -rmax_v, rmax_v).item()
        if val < best_val:
            best_val   = val
            best_state = {k: v.cpu().clone() for k, v in ebm.state_dict().items()}
        print(f'  ep {ep}  train={loss.item():.3e}  val={val:.3e}')

ebm.load_state_dict(best_state)
torch.save(ebm.state_dict(), 'results/stage23/ebm.pth')
for p in ebm.parameters():
    p.requires_grad_(False)
ebm.eval()

# Stage III: train denoiser 
denoiser = Denoiser().to(device)
den_opt  = optim.AdamW(denoiser.parameters(), lr=1e-3)
den_sch  = optim.lr_scheduler.CosineAnnealingLR(den_opt, T_max=30000, eta_min=1e-5)
mse_fn   = nn.MSELoss()

best_val, best_state = float('inf'), None
patience, bad = 300, 0

for ep in range(1, 30001):
    denoiser.train()
    den_opt.zero_grad()

    u0_tr    = denoiser(t_tr, y_tr)
    l_data   = mse_fn(u0_tr, y_tr)
    r0_tr    = u0_tr - f_tr
    rmax     = get_rmax(r0_tr)
    l_ebm    = ebm_nll(t_tr, r0_tr, ebm, -rmax, rmax)
    loss     = LAM1 * l_data + LAM2 * l_ebm
    loss.backward()
    torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 5.0)
    den_opt.step(); den_sch.step()

    denoiser.eval()
    with torch.no_grad():
        u0_va  = denoiser(t_va, y_va)
        l_dv   = mse_fn(u0_va, y_va)
        r0_va  = u0_va - f_va
        rmax_v = get_rmax(r0_va)
        l_ev   = ebm_nll(t_va, r0_va, ebm, -rmax_v, rmax_v)
        val    = (LAM1 * l_dv + LAM2 * l_ev).item()

    if val < best_val - 1e-6:
        best_val   = val
        best_state = {k: v.cpu().clone() for k, v in denoiser.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= patience:
            print(f'  early stop at ep {ep}'); break

    if ep % 2000 == 0:
        print(f'  ep {ep}  train={loss.item():.3e}  val={val:.3e}')

denoiser.load_state_dict(best_state)
torch.save(denoiser.state_dict(), 'results/stage23/denoiser.pth')

# evaluation
denoiser.eval()
rows = []

for sig in SIGMA_LEVELS:
    for seed in TEST_SEEDS:
        x_test = add_noise(x_true, sig, std_ref, seed=seed)
        with torch.no_grad():
            u0_te = denoiser(t_all[idx_te], x_test[idx_te])
        m = compute_metrics(x_true[idx_te].cpu().numpy(), u0_te.cpu().numpy())
        rows.append({'sigma': sig, 'seed': seed, **m})

# print metrics
for sig in SIGMA_LEVELS:
    sub = [r for r in rows if r['sigma'] == sig]
    for key in ('RMSE', 'R2', 'SNR_dB'):
        vals = np.array([r[key] for r in sub])
        print(f'  sigma={sig:.2f}  {key}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}')

# one representative plot
sig, seed = 0.25, TEST_SEEDS[0]
x_test = add_noise(x_true, sig, std_ref, seed=seed)
with torch.no_grad():
    u0_te = denoiser(t_all[idx_te], x_test[idx_te])

t_np  = t_all[idx_te].cpu().numpy().reshape(-1)
order = np.argsort(t_np)

plt.figure(figsize=(9, 4))
plt.plot(t_np[order], x_true[idx_te].cpu().numpy().reshape(-1)[order], label='true')
plt.plot(t_np[order], x_test[idx_te].cpu().numpy().reshape(-1)[order], alpha=0.4, label='noisy')
plt.plot(t_np[order], u0_te.cpu().numpy().reshape(-1)[order], label='denoised', lw=2)
plt.xlabel('t'); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('results/stage23/denoised_25pct.png')
plt.close()
print('Done.')