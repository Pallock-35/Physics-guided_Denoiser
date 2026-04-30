import numpy as np
import torch


def damped_sho(t, m=1.0, k=1.0, c=0.5):
    alpha = c / (2 * m)
    wd = np.sqrt(max(k / m - alpha**2, 1e-12))
    with torch.no_grad():
        u = torch.exp(-alpha * t) * (torch.cos(wd * t) + (alpha / wd) * torch.sin(wd * t))
    return u


def add_noise(u, noise_frac, std_ref, seed):
    torch.manual_seed(seed)
    return u + noise_frac * std_ref * torch.randn_like(u)


def split_indices(N, seed=2025, device='cpu'):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    perm = torch.randperm(N, generator=g, device=device)
    n_tr, n_va = int(0.7 * N), int(0.15 * N)
    return perm[:n_tr], perm[n_tr:n_tr+n_va], perm[n_tr+n_va:]


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2   = 1 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-12)
    snr  = 10 * np.log10(np.var(y_true) / (np.var(y_true - y_pred) + 1e-12))
    return {'RMSE': float(rmse), 'R2': float(r2), 'SNR_dB': float(snr)}


def ebm_nll(t, r, ebm, r_min, r_max, n_grid=401):
    N = t.shape[0]
    r_grid = torch.linspace(r_min, r_max, n_grid, device=t.device).view(1, n_grid, 1)
    dr = (r_max - r_min) / (n_grid - 1)

    t_exp = t.view(N, 1, 1).expand(N, n_grid, 1)
    r_exp = r_grid.expand(N, n_grid, 1)
    h_grid = ebm(t_exp.reshape(-1, 1), r_exp.reshape(-1, 1)).view(N, n_grid)

    log_Z  = torch.logsumexp(h_grid, dim=1) + np.log(dr)
    h_data = ebm(t, torch.clamp(r, r_min, r_max)).view(N)
    return torch.mean(log_Z - h_data)


def get_rmax(r, q=0.995):
    return float(max(torch.quantile(r.detach().abs(), q).item() + 1e-6, 0.3))