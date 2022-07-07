#!/usr/bin/env python

import copy
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
from src.parametrization import SPD, Diagonal, Fixed, Spherical
from src.dynamics_second import DynamicsSecond, ZeroVelocity
from src.dynamics_first import DynamicsFirst
from src.utils import squared_exp, infty_exp

# User input
dataset = sys.argv[1] if len(sys.argv) > 1 else "Angle"
obstacle = sys.argv[2].lower() in ['true', '1', 't', 'y', 'yes',
                                   'load'] if len(sys.argv) > 2 else False

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = "cpu"  # torch.device("cuda" if use_cuda else "cpu")

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

# Attractor
attractor = X[-1, :dim]

# Dynamics First
embedding_1 = Embedding(approximator)
embedding_1.apply(embedding_1.init_weights)

stiffness_1 = SPD(dim)
stiffness_1.vec_ = nn.Parameter(torch.tensor([1.0, 0.0]))
lambda_1 = -3
stiffness_1.eig_ = nn.Parameter(lambda_1 * torch.ones(2))

ds1 = DynamicsFirst(attractor, stiffness_1, embedding_1).to(device)
# ds1.load_state_dict(torch.load(os.path.join(
#     'models', '{}.pt'.format(dataset+"1")), map_location=torch.device(device)))
# ds1.eval()

# Dynamics Second
embedding_2 = copy.deepcopy(embedding_1)

stiffness_2 = SPD(dim)
stiffness_2.vec_ = nn.Parameter(torch.tensor([1.0, 0.0]))
lambda_2 = 0.5
stiffness_2.eig_ = nn.Parameter(lambda_2 * torch.ones(2))

dissipation = copy.deepcopy(stiffness_2)
lambda_3 = np.log(2) + lambda_2
dissipation.eig_ = nn.Parameter(lambda_3 * torch.ones(2))

ds = DynamicsSecond(attractor, stiffness_2,
                    dissipation, embedding_2).to(device)
# ds.velocity_ = ds1
# ds.velocity_ = ZeroVelocity()
# ds.load_state_dict(torch.load(os.path.join(
#     'models', '{}.pt'.format(dataset+"2")), map_location=torch.device(device)))
# ds.eval()

# Obstacle
a_obs, b_obs, eta = 1, 2, 1
r_obs = 0.05
theta = torch.linspace(-np.pi/5, 0.8*np.pi, 50).unsqueeze(1).to(device)
c = torch.tensor([[0.0000,   -0.1000]]).to(device)
r = 0.2
concave_obs = c + r*torch.cat((np.cos(theta), np.sin(theta)), axis=1)

x_obs = torch.tensor([[0.0000,   -0.1000]]).to(device)
y_obs = ds.embedding(x_obs)


# Metrics
def metric_exp(y):
    d = y-y_obs
    sigma = r_obs
    k = eta*torch.exp(-0.5*torch.sum(d.pow(2), dim=1) /
                      sigma ** 2).unsqueeze(1).unsqueeze(2)
    return torch.bmm(d.unsqueeze(2), d.unsqueeze(1)) * k.pow(2)/np.power(sigma, 4) + torch.eye(y.shape[1]).repeat(y.shape[0], 1, 1).to(y.device)


def metric_infty(y):
    d = y-y_obs
    k = (torch.norm(d, dim=1) - r_obs).unsqueeze(1).unsqueeze(2)
    k = torch.exp(a_obs/(b_obs*torch.pow(k, b_obs)))
    return torch.bmm(d.unsqueeze(2), d.unsqueeze(1)) * (k-1) + torch.eye(y.shape[1]).repeat(y.shape[0], 1, 1).to(y.device)


if obstacle:
    # ds.embedding.metric = metric_exp
    # ds.embedding.metric = metric_infty

    # ds.embedding.deformation = lambda x, y: infty_exp(
    #     x, y, r=0.005, a=0.005, b=2)
    ds.embedding.deformation = lambda x, y: squared_exp(
        x, y, sigma=0.05, eta=10)
    ds.embedding.obstacles = x_obs

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
# start_point = x_train[1005, :]
start_point = torch.tensor([-0.5, -0.66])
a = [start_point[0] - box_side, start_point[1] - box_side]
b = [start_point[0] + box_side, start_point[1] + box_side]

T = 5
dt = 0.001
steps = int(np.ceil(T/dt))
num_samples = 1

samples = np.zeros((steps, num_samples, 2*dim))
samples[0, :, 0] = np.random.uniform(a[0], b[0], num_samples)
samples[0, :, 1] = np.random.uniform(a[1], b[1], num_samples)

samples[0, 0, :dim] = np.array([-0.49933788, -0.67346152])
samples[0, 0, dim:] = 1 * \
    (attractor.cpu().detach().numpy() - samples[0, 0, :dim])

for step in range(steps-1):
    state = torch.from_numpy(samples[step, :, :]).float().to(
        device).requires_grad_(True)
    samples[step+1, :, dim:] = samples[step, :, dim:] + \
        dt*ds(state).cpu().detach().numpy()
    samples[step+1, :, :dim] = samples[step,
                                       :, :dim] + dt*samples[step+1, :, dim:]

# Vector Field
field = X_test[:, dim:] + dt * ds(X_test)
field = field.cpu().detach().numpy()
x_field = field[:, 0].reshape(resolution, -1, order="F")
y_field = field[:, 1].reshape(resolution, -1, order="F")

# Plot potential contour
fig = plt.figure()
ax = fig.add_subplot(111)
fig.colorbar(mappable,  ax=ax, label=r"$\phi$")

ax.contourf(x_mesh, y_mesh, phi, 500, cmap="jet")
ax.contour(x_mesh, y_mesh, phi, 10, cmap=None, colors='#f2e68f')
ax.scatter(x_train[::10, 0], x_train[::10, 1],
           s=20, edgecolors='k', c='red')

ax.axis('square')
ax.set_xlabel('$x^1$')
ax.set_ylabel('$x^2$')

# Plot embedding
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
fig.colorbar(mappable,  ax=ax, label=r"$\phi$")

ax.plot_surface(x_embedding, y_embedding, z_embedding,
                facecolors=colors, antialiased=True, linewidth=0, alpha=0.5)
ax.scatter(train_embedding[::10, 0], train_embedding[::10, 1],
           train_embedding[::10, 2], s=20, edgecolors='k', c='red')
if obstacle:
    y_obs = y_obs.cpu().detach().numpy()
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    obs_x = y_obs[0, 0] + r_obs*np.cos(u)*np.sin(v)
    obs_y = y_obs[0, 1] + r_obs*np.sin(u)*np.sin(v)
    obs_z = y_obs[0, 2] + 0.6*np.cos(v)
    ax.plot_surface(obs_x, obs_y, obs_z, linewidth=0.0,
                    cstride=1, rstride=1)

# ax.set_box_aspect((np.ptp(x_embedding), np.ptp(
#     y_embedding), np.ptp(z_embedding)))
ax.set_xlabel('$y^1$')
ax.set_ylabel('$y^2$')
ax.set_zlabel('$y^3$')

# Plot vector field
fig = plt.figure()
ax = fig.add_subplot(111)
fig.colorbar(mappable,  ax=ax, label=r"$\phi$")

ax.streamplot(x_mesh, y_mesh, x_field, y_field, color=phi, cmap="jet")
for i in range(num_samples):
    ax.plot(samples[:, i, 0], samples[:, i, 1], color='k')
rect = patches.Rectangle((start_point[0] - box_side, start_point[1] - box_side),
                         2*box_side, 2*box_side, linewidth=1, edgecolor='k', facecolor='none')
ax.add_patch(rect)
ax.scatter(x_train[::10, 0], x_train[::10, 1],
           s=20, edgecolors='k', c='red')
ax.scatter(x_train[-1, 0], x_train[-1, 1], s=100,
           edgecolors='k', c='yellow', marker="*")
if obstacle:
    x_obs = x_obs.cpu().detach().numpy()
    circ = plt.Circle((x_obs[0, 0], x_obs[0, 1]), r_obs,
                      color='k', fill='grey', alpha=0.5)
    ax.add_patch(circ)
    # concave_obs = concave_obs.cpu().detach().numpy()
    # ax.scatter(concave_obs[:, 0], concave_obs[:, 1])

ax.axis('square')
ax.set_xlabel('$x^1$')
ax.set_ylabel('$x^2$')

plt.show()
