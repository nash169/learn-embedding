#!/usr/bin/env python

import copy
import os
import sys
from more_itertools import sample
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.kernel_machine import KernelMachine
from src.coupling_layer import CouplingLayer
from src.feedforward import FeedForward
from src.embedding import Embedding
from src.parametrization import SPD, Diagonal, Fixed, Spherical
from src.dynamics_second import DynamicsSecond, ZeroVelocity
from src.dynamics_first import DynamicsFirst

# User input
dataset = sys.argv[1] if len(sys.argv) > 1 else "Angle"
obstacle = sys.argv[2].lower() in ['true', '1', 't', 'y', 'yes',
                                   'load'] if len(sys.argv) > 2 else False

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data
data = np.loadtxt(os.path.join('trainset', '{}.csv'.format(dataset)))
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
# approximator = KernelMachine(dim, 500, 1, length=0.3)
approximator = FeedForward(dim, [64], 1)
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
stiffness = SPD(dim)

# Dynamics First
ds1 = DynamicsFirst(attractor, stiffness, embedding).to(device)
ds1.load_state_dict(torch.load(os.path.join(
    'models', '{}.pt'.format(dataset+"1")), map_location=torch.device(device)))
ds1.eval()

# Dynamics Second
dissipation = copy.deepcopy(stiffness)
dissipation.eig_ = nn.Parameter(2*torch.sqrt(dissipation.eig_))

# dissipation = SPD(dim)

# L, V = torch.linalg.eig(stiffness.spd.weight.data)
# dissipation.fixed = torch.mm(torch.real(V.transpose(1, 0)), torch.mm(torch.diag(
#     2*torch.sqrt(torch.real(L))), torch.real(V)))

ds = DynamicsSecond(attractor, stiffness,
                    dissipation, embedding).to(device)
ds.velocity_ = ds1
ds.load_state_dict(torch.load(os.path.join(
    'models', '{}.pt'.format(dataset+"2")), map_location=torch.device(device)))
ds.eval()
# ds.velocity_ = ds1
# ds.velocity_ = ZeroVelocity()


# Obstacle
a_obs, b_obs, eta = 1, 2, 1
r = 0.05
# x_obs = torch.tensor([[-0.3000,   0.0000]]).to(device)
x_obs = torch.tensor([[0.0000,   -0.1000]]).to(device)
y_obs = ds.embedding(x_obs)


# Metrics
def metric_exp(y):
    d = y-y_obs
    sigma = r
    k = eta*torch.exp(-0.5*torch.sum(d.pow(2), dim=1) /
                      sigma ** 2).unsqueeze(1).unsqueeze(2)
    return torch.bmm(d.unsqueeze(2), d.unsqueeze(1)) * k.pow(2)/np.power(sigma, 4) + torch.eye(y.shape[1]).repeat(y.shape[0], 1, 1).to(y.device)


def metric_infty(y):
    d = y-y_obs
    k = (torch.norm(d, dim=1) - r).unsqueeze(1).unsqueeze(2)
    k = torch.exp(a_obs/(b_obs*torch.pow(k, b_obs)))
    return torch.bmm(d.unsqueeze(2), d.unsqueeze(1)) * (k-1) + torch.eye(y.shape[1]).repeat(y.shape[0], 1, 1).to(y.device)


if obstacle:
    ds.embedding.metric = metric_exp
    ds.embedding.metric = metric_infty
    ds.embedding.obstacles = torch.tensor([[0.0000,   -0.1000]]).to(device)

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
box_side = 0.03
a = [x_train[0, 0] - box_side, x_train[0, 1] - box_side]
b = [x_train[0, 0] + box_side, x_train[0, 1] + box_side]

T = 5
dt = 0.001
steps = int(np.ceil(T/dt))
num_samples = 3

samples = np.zeros((steps, num_samples, 2*dim))
samples[0, :, 0] = np.random.uniform(a[0], b[0], num_samples)
samples[0, :, 1] = np.random.uniform(a[1], b[1], num_samples)

for step in range(steps-1):
    state = torch.from_numpy(samples[step, :, :]).float().to(
        device).requires_grad_(True)
    samples[step+1, :, dim:] = samples[step, :, dim:] + \
        dt*ds(state).cpu().detach().numpy()
    samples[step+1, :, :dim] = samples[step,
                                       :, :dim] + dt*samples[step+1, :, dim:]

# samples = []
# for i in range(num_samples):
#     state = np.zeros([steps, 2*dim])
#     state[0, 0] = np.random.uniform(a[0], b[0])
#     state[0, 1] = np.random.uniform(a[1], b[1])
#     samples.append(state)

# samples[0][0, :dim] = x_train[0, :]
# samples[0][0, dim:] = dx_train[0, :]

# for step in range(steps-1):
#     for i in range(num_samples):
#         X_sample = torch.from_numpy(samples[i][step, :]).float().to(
#             device).requires_grad_(True).unsqueeze(0)
#         samples[i][step+1, dim:] = samples[i][step, dim:] + \
#             dt*ds(X_sample).cpu().detach().numpy()
#         samples[i][step+1, :dim] = samples[i][step, :dim] + \
#             dt*samples[i][step+1, dim:]

# Vector Field
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
    obs_z = y_obs[0, 2] + 0.6*np.cos(v)
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
    ax.plot(samples[:, i, 0], samples[:, i, 1], color='k')
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
