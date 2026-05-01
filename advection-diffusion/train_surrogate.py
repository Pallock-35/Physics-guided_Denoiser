import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs("results/stage1", exist_ok=True)

# config
X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 0.1
K       = 0.0025   # diffusion coefficient (known)
SIGMA0  = 0.06
X0      = 0.25
EPOCHS  = 2500
LR      = 2e-3

# training points
rng = np.random.default_rng(0)
tt  = lambda a: torch.tensor(a, dtype=torch.float32, device=device)

# collocation (PDE residual)
N_f = 5000
x_f = rng.uniform(X_MIN, X_MAX, (N_f, 1))
t_f = rng.uniform(T_MIN, T_MAX, (N_f, 1))
x_f_t, t_f_t = tt(x_f), tt(t_f)

# initial condition
N_ic = 200
x_ic = np.linspace(X_MIN, X_MAX, N_ic)[:, None]
t_ic = np.zeros_like(x_ic)
u_ic = np.exp(-((x_ic - X0) ** 2) / SIGMA0 ** 2).astype(np.float32)
x_ic_t, t_ic_t, u_ic_t = tt(x_ic), tt(t_ic), tt(u_ic)

# boundary conditions
N_bc = 200
t_bc = rng.uniform(T_MIN, T_MAX, (N_bc, 1))
x_bc0_t = tt(np.full_like(t_bc, X_MIN))
x_bc1_t = tt(np.full_like(t_bc, X_MAX))
t_bc_t  = tt(t_bc)
u_bc_t  = tt(np.zeros((N_bc, 1), dtype=np.float32))

# model
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

model = PINN_Diff().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=LR)
mse   = nn.MSELoss()


def diffusion_residual(x, t):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    u    = model(x, t)
    u_t  = torch.autograd.grad(u,   t, torch.ones_like(u),   create_graph=True)[0]
    u_x  = torch.autograd.grad(u,   x, torch.ones_like(u),   create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t - K * u_xx   # should equal zero


# training
model.train()
for ep in range(1, EPOCHS + 1):
    opt.zero_grad()
    l_pde = mse(diffusion_residual(x_f_t, t_f_t), torch.zeros(N_f, 1, device=device))
    l_ic  = mse(model(x_ic_t, t_ic_t), u_ic_t)
    l_bc  = mse(model(x_bc0_t, t_bc_t), u_bc_t) + mse(model(x_bc1_t, t_bc_t), u_bc_t)
    loss  = l_pde + 10.0 * l_ic + l_bc
    loss.backward()
    opt.step()

    if ep % 500 == 0:
        print(f"  ep {ep}  loss={loss.item():.3e}  pde={l_pde.item():.3e}  ic={l_ic.item():.3e}")

# save
torch.save({"state": model.state_dict()}, "results/stage1/surrogate.pth")


model.eval()
x_plot = np.linspace(X_MIN, X_MAX, 200)[:, None].astype(np.float32)
t_plot = np.full_like(x_plot, 0.05)
with torch.no_grad():
    u_plot = model(tt(x_plot), tt(t_plot)).cpu().numpy().reshape(-1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_plot, u_plot, label="surrogate f(x, t=0.05)")
ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/stage1/surrogate_slice.png", dpi=200)
plt.close(fig)
print("Done.")
