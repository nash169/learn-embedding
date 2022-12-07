#!/usr/bin/env python

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.utils import create_model
from src.utils import squared_exp, infty_exp

# User input
dataset = sys.argv[1] if len(sys.argv) > 1 else "RobotDemo"
obstacle = sys.argv[2].lower() in ['true', '1', 't', 'y', 'yes',
                                   'load'] if len(sys.argv) > 2 else False

# CPU/GPU setting
use_cuda = False  # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data
data = np.loadtxt(os.path.join('data/train', '{}.csv'.format(dataset)))
dim = int(data.shape[1]/3)

# Tensor data
X = torch.from_numpy(data[:, :2*dim]).float().to(device).requires_grad_(True)
Y = torch.from_numpy(data[:, 2*dim:]).float().to(device).requires_grad_(True)

# Plot data
x_train = data[:, :dim]
dx_train = data[:, dim:2*dim]
ddx_train = data[:, 2*dim:]

# Model
load = True
order = "first"
ds = create_model(X, order)
if load:
    ds.load_state_dict(torch.load(os.path.join(
        'models', '{}.pt'.format(dataset+type(ds).__name__)), map_location=torch.device(device)))
    ds.eval()

# Sampled Dynamics
T = 20
dt = 0.01
num_samples = 1
init_state = torch.cat((X[2000, :dim].unsqueeze(0),
                       torch.zeros((num_samples, dim))), dim=1)
samples = ds.integrate(init_state, T, dt)

# Plot vector field
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for i in range(num_samples):
    ax.plot(samples.detach().numpy()[:, i, 0],
            samples.detach().numpy()[:, i, 1], samples.detach().numpy()[:, i, 2], color='k')
ax.scatter(x_train[::10, 0], x_train[::10, 1], x_train[::10, 2],
           s=20, edgecolors='k', c='red')
ax.scatter(x_train[-1, 0], x_train[-1, 1], x_train[-1, 2], s=100,
           edgecolors='k', c='yellow', marker="*")

# ax.axis('square')
ax.set_xlabel('$x^1$')
ax.set_ylabel('$x^2$')
ax.set_ylabel('$x^3$')

plt.show()
