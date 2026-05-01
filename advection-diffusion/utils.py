import numpy as np
import torch

def advdiff_fd(x, t, u0, k, c):
    Nx, Nt = len(x), len(t)
    dx, dt = x[1] - x[0], t[1] - t[0]
    r_diff = k * dt / dx**2
    r_adv  = abs(c) * dt / dx
    print(f"FD checks: r_diff={r_diff:.4f} (<=0.5), r_adv={r_adv:.4f} (<=1)")

    U = np.zeros((Nt, Nx), dtype=np.float64)
    U[0] = u0.copy()
    U[0, 0] = U[0, -1] = 0.0

    for n in range(Nt - 1):
        u = U[n]
        u_xx = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        u_x  = (u[1:-1] - u[:-2]) / dx if c >= 0 else (u[2:] - u[1:-1]) / dx
        U[n+1, 1:-1] = u[1:-1] + dt * (k * u_xx - c * u_x)
        U[n+1, 0] = U[n+1, -1] = 0.0

    return U



# Noise
def add_noise(u_flat, noise_frac, amp, seed):
    """Add Gaussian noise scaled to amp (max of true signal)."""
    rng = np.random.default_rng(int(seed))
    return u_flat + noise_frac * amp * rng.standard_normal(size=u_flat.shape)


# Metrics
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    err  = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    r2   = float(1 - np.sum(err**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-12))
    snr  = float(10 * np.log10(np.var(y_true) / (np.var(err) + 1e-12)))
    return {"RMSE": rmse, "R2": r2, "SNR_dB": snr}



# EBM NLL
def ebm_nll(x, t, r, ebm, r_min, r_max, n_grid=201):
    device = x.device
    N  = x.shape[0]
    r  = torch.clamp(r, r_min, r_max)
    dr = (r_max - r_min) / (n_grid - 1)

    r_grid = torch.linspace(r_min, r_max, n_grid, device=device).view(1, n_grid, 1)
    x_exp  = x.view(N, 1, 1).expand(N, n_grid, 1)
    t_exp  = t.view(N, 1, 1).expand(N, n_grid, 1)
    rg_exp = r_grid.expand(N, n_grid, 1)

    h_grid = ebm(x_exp.reshape(-1, 1),
                 t_exp.reshape(-1, 1),
                 rg_exp.reshape(-1, 1)).view(N, n_grid)

    log_Z  = torch.logsumexp(h_grid, dim=1) + np.log(dr)
    h_data = ebm(x, t, r).view(N)
    return torch.mean(log_Z - h_data)


@torch.no_grad()
def get_rmax(r, q=0.995, floor=0.25, cap=5.0):
    rmax = torch.quantile(r.detach().abs().reshape(-1), q).item() + 1e-6
    return float(min(max(rmax, floor), cap))
