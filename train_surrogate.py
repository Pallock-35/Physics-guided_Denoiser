import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models import Surrogate

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('results/stage1', exist_ok=True)

# ---- data ----
t0, t1 = 0.0, 15.0
t_col = torch.FloatTensor(10000, 1).uniform_(t0, t1).to(device)
t_ic  = torch.tensor([[0.0]], device=device)
x_ic  = torch.tensor([[1.0]], device=device)
v_ic  = torch.tensor([[0.0]], device=device)

t_val    = torch.linspace(t0, t1, 2000, device=device).view(-1, 1)
x_val_np = np.cos(t_val.cpu().numpy())

# ---- model ----
model = Surrogate().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
mse   = nn.MSELoss()


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y),
                                create_graph=True, retain_graph=True)[0]


def pde_loss():
    t = t_col.clone().requires_grad_(True)
    x = model(t)
    x_tt = grad(grad(x, t), t)
    return mse(x_tt + x, torch.zeros_like(x))


def ic_loss():
    t0_ = t_ic.clone().requires_grad_(True)
    x0  = model(t0_)
    v0  = grad(x0, t0_)
    return mse(x0, x_ic) + mse(v0, v_ic)


# ---- training ----
for ep in range(1, 20001):
    opt.zero_grad()
    loss = pde_loss() + ic_loss()
    loss.backward()
    opt.step()

    if ep % 2000 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(t_val).cpu().numpy()
        rmse = np.sqrt(np.mean((pred - x_val_np)**2))
        print(f'ep {ep}  loss={loss.item():.3e}  val RMSE={rmse:.4e}')
        model.train()

# ---- save ----
torch.save({'state': model.state_dict()}, 'results/stage1/surrogate.pth')

model.eval()
with torch.no_grad():
    pred = model(t_val).cpu().numpy().reshape(-1)

t_np = t_val.cpu().numpy().reshape(-1)
plt.figure(figsize=(9, 4))
plt.plot(t_np, x_val_np.reshape(-1), label='cos(t)')
plt.plot(t_np, pred, '--', label='surrogate')
plt.xlabel('t'); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('results/stage1/surrogate_fit.png')
plt.close()
print('Done')