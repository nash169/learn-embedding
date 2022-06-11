#!/usr/bin/env python

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.kernel_machine import KernelMachine
from src.coupling_layer import CouplingLayer
from src.feedforward import FeedForward
from src.embedding import Embedding
from src.parametrization import SPD, Diagonal, Spherical
from src.dynamics_second import DynamicsSecond
from src.dynamics_first import DynamicsFirst

# User input
dataset = sys.argv[1] if len(sys.argv) > 1 else "Angle"
obstacle = sys.argv[2].lower() in ['true', '1', 't', 'y', 'yes',
                                   'load'] if len(sys.argv) > 2 else False
first = True

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data
data = np.loadtxt(os.path.join('data', '{}.csv'.format(dataset)))
dim = int(data.shape[1]/3)

# Tensor data
X = torch.from_numpy(data[:, :2*dim]).float().to(device).requires_grad_(True)
Y = torch.from_numpy(data[:, 2*dim:]).float().to(device).requires_grad_(True)

# Plot data
x_train = data[:, :dim]
dx_train = data[:, dim:2*dim]
ddx_train = data[:, 2*dim:]

# Test data
resolution = 100
lower, upper = -1, 1
x_mesh, y_mesh = np.meshgrid(np.linspace(lower, upper, resolution),
                             np.linspace(lower, upper, resolution))
x_test = np.array(
    [x_mesh.ravel(order="F"), y_mesh.ravel(order="F")]).transpose()
X_test = np.zeros([x_test.shape[0], 2*x_test.shape[1]])
X_test[:, :x_test.shape[1]] = x_test

x_test = torch.from_numpy(
    x_test).float().to(device).requires_grad_(True)
X_test = torch.from_numpy(
    X_test).float().to(device).requires_grad_(True)

# Function approximator
approximator = KernelMachine(dim, 1000, 1, length=0.2)
# approximator = FeedForward(dim, [128, 128, 128], 1, 3)
# layers = nn.ModuleList()
# layers.append(KernelMachine(dim, 250, dim+1, length=0.45))
# for i in range(2):
#     layers.append(CouplingLayer(dim+1, 250, i % 2, 0.45))
# approximator = nn.Sequential(*(layers[i] for i in range(len(layers))))

# Embedding
embedding = Embedding(approximator)

# Attractor
attractor = X[-1, :dim]

# Stiffness
stiffness = Diagonal(dim)

# Dissipation
dissipation = Diagonal(dim)

# Dynamics
if first:
    ds = DynamicsFirst(attractor, stiffness, embedding).to(device)
else:
    ds = DynamicsSecond(attractor, stiffness,
                        dissipation, embedding).to(device)

# Load dict
ds.load_state_dict(torch.load(os.path.join(
    'models', '{}.pt'.format(dataset)), map_location=torch.device(device)))
ds.eval()

# Obstacle
a_obs, b_obs, eta = 1, 4, 1000
r = 0.1
x_obs = torch.tensor([[-0.3000,   0.0000]]).to(device)
y_obs = ds.embedding(x_obs)
if obstacle:
    # def metric(y):
    #     n = 1 + eta*torch.exp(-0.5*torch.norm(y-y_obs, dim=1) /
    #                           r**2).unsqueeze(1).unsqueeze(2)
    #     return torch.eye(y.shape[1]).repeat(y.shape[0], 1, 1).to(device)*n

    # def metric(y):
    #     d = (torch.norm(y-y_obs, dim=1) + r+0.05).unsqueeze(1).unsqueeze(2) + 1
    #     dd = 0.5*(y-y_obs)/torch.norm(y-y_obs, dim=1).unsqueeze(1)
    #     return torch.bmm(dd.unsqueeze(2), dd.unsqueeze(1)) * torch.exp(a_obs/(b_obs*torch.pow(d, b_obs))) + 0.01*torch.eye(y.shape[1]).repeat(y.shape[0], 1, 1).to(y.device)

    def metric(y):
        d = y-y_obs
        k = eta*torch.exp(-0.5*torch.norm(d, dim=1)/r **
                          2).unsqueeze(1).unsqueeze(2)
        return torch.bmm(d.unsqueeze(2), d.unsqueeze(1)) * k + torch.eye(y.shape[1]).repeat(y.shape[0], 1, 1).to(y.device)
else:
    def metric(y):
        g = torch.eye(y.shape[1])
        return g.repeat(y.shape[0], 1, 1).to(device)
ds.embedding.metric = metric

# Potential
phi = ds.potential(x_test)
phi = phi.cpu().detach().numpy()
phi = phi.reshape(resolution, -1, order="F")
phi -= np.min(phi)
phi /= np.max(phi)
colors = plt.cm.jet(phi)
mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
mappable.set_array(phi)

# Embedding
test_embedding = ds.embedding(x_test)
test_embedding = test_embedding.cpu().detach().numpy()
x_embedding = test_embedding[:, 0].reshape(resolution, -1, order="F")
y_embedding = test_embedding[:, 1].reshape(resolution, -1, order="F")
z_embedding = test_embedding[:, 2].reshape(resolution, -1, order="F")
train_embedding = ds.embedding(X[:, :dim]).cpu().detach().numpy()

# Sampled Dynamics
box_side = 0.05
a = [x_train[0, 0] - box_side, x_train[0, 1] - box_side]
b = [x_train[0, 0] + box_side, x_train[0, 1] + box_side]

T = 5
dt = 0.01
steps = int(np.ceil(T/dt))
num_samples = 3
samples = []

for i in range(num_samples):
    state = np.zeros([steps, 2*dim])
    state[0, 0] = np.random.uniform(a[0], b[0])
    state[0, 1] = np.random.uniform(a[1], b[1])
    samples.append(state)

if not first:
    for step in range(steps-1):
        for i in range(num_samples):
            X_sample = torch.from_numpy(samples[i][step, :]).float().to(
                device).requires_grad_(True).unsqueeze(0)
            samples[i][step+1, dim:] = samples[i][step, dim:] + \
                dt*ds(X_sample).cpu().detach().numpy()
            samples[i][step+1, :dim] = samples[i][step, :dim] + \
                dt*samples[i][step+1, dim:]
else:
    for step in range(steps-1):
        for i in range(num_samples):
            X_sample = torch.from_numpy(samples[i][step, :dim]).float().to(
                device).requires_grad_(True).unsqueeze(0)
            samples[i][step+1, dim:] = ds(X_sample).cpu().detach().numpy()
            samples[i][step+1, :dim] = samples[i][step, :dim] + \
                dt*samples[i][step+1, dim:]

# Vector Field
if first:
    field = ds(X_test[:, :dim])
else:
    field = X_test[:, dim:] + dt * ds(X_test)
field = field.cpu().detach().numpy()
x_field = field[:, 0].reshape(resolution, -1, order="F")
y_field = field[:, 1].reshape(resolution, -1, order="F")

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
contour = ax.contourf(x_mesh, y_mesh, phi, 500, cmap="jet")
ax.contour(x_mesh, y_mesh, phi, 10, cmap=None, colors='#f2e68f')
fig.colorbar(mappable,  ax=ax, label=r"$\phi$")
ax.scatter(x_train[::10, 0], x_train[::10, 1],
           s=20, edgecolors='k', c='red')
ax.axis('square')

ax.set_xlabel('$x^1$')
ax.set_ylabel('$x^2$')

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x_embedding, y_embedding, z_embedding,
                facecolors=colors, antialiased=True, linewidth=0, alpha=0.5)
fig.colorbar(mappable,  ax=ax, label=r"$\phi$")
# ax.set_box_aspect((np.ptp(x_embedding), np.ptp(
#     y_embedding), np.ptp(z_embedding)))
ax.scatter(train_embedding[::10, 0], train_embedding[::10, 1],
           train_embedding[::10, 2], s=20, edgecolors='k', c='red')
# Obstacle
if obstacle:
    y_obs = y_obs.cpu().detach().numpy()
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    obs_x = y_obs[0, 0] + r*np.cos(u)*np.sin(v)
    obs_y = y_obs[0, 1] + r*np.sin(u)*np.sin(v)
    obs_z = y_obs[0, 2] + 4.5*np.cos(v)
    ax.plot_surface(obs_x, obs_y, obs_z, linewidth=0.0,
                    cstride=1, rstride=1)

ax.set_xlabel('$y^1$')
ax.set_ylabel('$y^2$')
ax.set_zlabel('$y^3$')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.streamplot(x_mesh, y_mesh, x_field, y_field, color=phi, cmap="jet")
ax.axis('square')
fig.colorbar(mappable,  ax=ax, label=r"$\phi$")

for i in range(num_samples):
    ax.plot(samples[i][:, 0], samples[i][:, 1], color='k')
rect = patches.Rectangle((x_train[0, 0] - box_side, x_train[0, 1] - box_side),
                         2*box_side, 2*box_side, linewidth=1, edgecolor='k', facecolor='none')
ax.add_patch(rect)

ax.scatter(x_train[::10, 0], x_train[::10, 1],
           s=20, edgecolors='k', c='red')
ax.scatter(x_train[-1, 0], x_train[-1, 1], s=100,
           edgecolors='k', c='yellow', marker="*")

if obstacle:
    x_obs = x_obs.cpu().detach().numpy()
    circ = plt.Circle((x_obs[0, 0], x_obs[0, 1]), r,
                      color='k', fill='grey', alpha=0.5)
    ax.add_patch(circ)

ax.set_xlabel('$x^1$')
ax.set_ylabel('$x^2$')

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(x_train[::10, 0], x_train[::10, 1], s=20, edgecolors='k', c='red')
# ax.quiver()

plt.show()
