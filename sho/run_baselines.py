import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import damped_sho, add_noise, split_indices, compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('results/baselines', exist_ok=True)

T0, T1, N    = 0.0, 15.0, 5000
SIGMA_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25]
TEST_SEEDS   = [111, 222, 333]

t_all  = torch.linspace(T0, T1, N, device=device).view(-1, 1)
x_true = damped_sho(t_all).to(device)
std_ref = float(x_true.std())

idx_tr, idx_va, idx_te = split_indices(N, device=device)


# Gaussian filter
def gaussian_filter(x, sigma=1.0):
    x = x.view(1, 1, -1)
    r = int(3 * sigma + 0.5)
    xs = torch.arange(-r, r+1, device=x.device, dtype=x.dtype)
    k  = torch.exp(-0.5 * (xs / sigma)**2)
    k  = (k / k.sum()).view(1, 1, -1)
    pad = k.shape[-1] // 2
    return nn.functional.conv1d(nn.functional.pad(x, (pad, pad), 'reflect'), k).view(-1)


# Wavelet denoising
def wavelet_denoise(x_np):
    try:
        import pywt
    except ImportError:
        return None
    x_np = x_np.reshape(-1)
    coeffs = pywt.wavedec(x_np, 'haar', level=1)
    thr = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(x_np)))
    coeffs[1:] = [pywt.threshold(c, thr, 'soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, 'haar')[:len(x_np)]


# PINN-noisy 
class PINNNoisy(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(1, 64), nn.Tanh()]
        for _ in range(5):
            layers += [nn.Linear(64, 64), nn.Tanh()]
        layers += [nn.Linear(64, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, t):
        return self.net(t)


def train_pinn_noisy(sig_frac, seed):
    torch.manual_seed(seed)
    x_obs = add_noise(x_true, sig_frac, std_ref, seed=seed)

    t_tr_ = t_all[idx_tr]; y_tr_ = x_obs[idx_tr]
    t_va_ = t_all[idx_va]; y_va_ = x_obs[idx_va]

    t_col = (T0 + (T1 - T0) * torch.rand(4000, 1, device=device))

    model = PINNNoisy().to(device)
    opt   = optim.AdamW(model.parameters(), lr=1e-3)
    mse   = nn.MSELoss()

    def pde_res(t):
        t = t.clone().requires_grad_(True)
        x = model(t)
        dx  = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
        d2x = torch.autograd.grad(dx, t, torch.ones_like(dx), create_graph=True)[0]
        return d2x + x

    best_val, best_state, bad = float('inf'), None, 0

    for ep in range(1, 4001):
        model.train(); opt.zero_grad()
        loss = mse(model(t_tr_), y_tr_) + torch.mean(pde_res(t_col)**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if ep % 100 == 0:
            model.eval()
            with torch.no_grad():
                vd = mse(model(t_va_), y_va_).item()
            with torch.enable_grad():
                vp = torch.mean(pde_res(t_va_)**2).item()
            v = vd + vp
            if v < best_val - 1e-5:
                best_val = v
                best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 25: break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        return model(t_all[idx_te]).cpu().numpy().reshape(-1)

rows = []

for sig in SIGMA_LEVELS:
    print(f'sigma={sig:.2f}')
    for seed in TEST_SEEDS:
        x_obs = add_noise(x_true, sig, std_ref, seed=seed)
        y_true_te = x_true[idx_te].cpu().numpy().reshape(-1)
        te_idx    = idx_te.cpu().numpy()

        # Gaussian
        xg = gaussian_filter(x_obs.view(-1)).detach().cpu().numpy()[te_idx]
        # Wavelet
        xw = wavelet_denoise(x_obs.cpu().numpy())
        xw = xw[te_idx] if xw is not None else None
        # PINN-noisy
        xp = train_pinn_noisy(sig, seed)

        for name, pred in [('Gaussian', xg), ('Wavelet', xw), ('PINN_noisy', xp)]:
            if pred is None: continue
            m = compute_metrics(y_true_te, pred)
            rows.append({'sigma': sig, 'seed': seed, 'method': name, **m})

# summary
for sig in SIGMA_LEVELS:
    for method in ('Gaussian', 'Wavelet', 'PINN_noisy'):
        sub = [r for r in rows if r['sigma'] == sig and r['method'] == method]
        if not sub: continue
        for key in ('RMSE', 'R2', 'SNR_dB'):
            vals = np.array([r[key] for r in sub])
            print(f'  sigma={sig:.2f}  {method:12s}  {key}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}')

# save csv
with open('results/baselines/metrics.csv', 'w') as f:
    f.write('sigma,seed,method,RMSE,R2,SNR_dB\n')
    for r in rows:
        f.write(f"{r['sigma']},{r['seed']},{r['method']},{r['RMSE']:.6f},{r['R2']:.6f},{r['SNR_dB']:.4f}\n")

print('Done.')