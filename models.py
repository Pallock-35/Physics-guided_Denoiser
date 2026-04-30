import torch
import torch.nn as nn


def make_mlp(in_dim, out_dim, width, depth, act=nn.Tanh):
    layers = [nn.Linear(in_dim, width), act()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), act()]
    layers += [nn.Linear(width, out_dim)]
    return nn.Sequential(*layers)


class Surrogate(nn.Module):
    # Stage I: trained on undamped SHO physics
    def __init__(self):
        super().__init__()
        self.net = make_mlp(1, 1, width=64, depth=6)

    def forward(self, t):
        return self.net(t)


class EBM(nn.Module):
    # Stage II: energy function h(t, r)
    def __init__(self):
        super().__init__()
        self.net = make_mlp(2, 1, width=32, depth=3)

    def forward(self, t, r):
        return self.net(torch.cat([t, r], dim=1))


class Denoiser(nn.Module):
    # Stage III: reconstructed state
    def __init__(self):
        super().__init__()
        self.net = make_mlp(2, 1, width=32, depth=4)

    def forward(self, t, y):
        return self.net(torch.cat([t, y], dim=1))