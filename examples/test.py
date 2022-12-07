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
dataset = sys.argv[1] if len(sys.argv) > 1 else "Khamesh"
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

# Test data
resolution = 100
grid = [x_train[:, 0].min(), x_train[:, 0].max(),
        x_train[:, 1].min(), x_train[:, 1].max()]
y_m, x_m = torch.meshgrid(torch.linspace(x_train[:, 0].min(), x_train[:, 0].max(), resolution),
                          torch.linspace(x_train[:, 1].min(), x_train[:, 1].max(), resolution))
X_test = torch.cat((torch.ravel(x_m).unsqueeze(-1),
                    torch.ravel(y_m).unsqueeze(-1), torch.zeros((resolution**2, dim))), dim=1).requires_grad_(True)

# Model
load = True
order = "first"
ds = create_model(X, order)
if load:
    ds.load_state_dict(torch.load(os.path.join(
        'models', '{}.pt'.format(dataset+type(ds).__name__)), map_location=torch.device(device)))
    ds.eval()

# Obstacle
if obstacle:
    theta = torch.linspace(-np.pi/5, 0.8*np.pi, 50).unsqueeze(1).to(device)
    c = torch.tensor([[0.0000,   -0.1000]]).to(device)
    r = 0.2
    x_obs = c + r*torch.cat((np.cos(theta), np.sin(theta)), axis=1)
    # x_obs = torch.tensor([[0.0000,   -0.1000]]).to(device)

    # ds.embedding.metric = metric_exp
    # ds.embedding.metric = metric_infty

    r_obs = 0.1
    # ds.embedding.deformation = lambda x, y: infty_exp(
    #     x, y, r=0.005, a=0.005, b=2)
    ds.embedding.deformation = lambda x, y: squared_exp(
        x, y, sigma=0.03, eta=10)

    ds.embedding.obstacles = x_obs
    y_obs = ds.embedding(x_obs)

# Potential
phi = ds.potential(X_test[:, :dim]).reshape(
    resolution, -1).cpu().detach().numpy()
phi -= np.min(phi)
phi /= np.max(phi)
colors = plt.cm.jet(phi)
mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
mappable.set_array(phi)

# Embedding
test_embedding = ds.embedding(X_test[:, :dim])
x_embedding = test_embedding[:, 0].reshape(
    resolution, -1).cpu().detach().numpy()
y_embedding = test_embedding[:, 1].reshape(
    resolution, -1).cpu().detach().numpy()
z_embedding = test_embedding[:, 2].reshape(
    resolution, -1).cpu().detach().numpy()
train_embedding = ds.embedding(X[:, :dim]).cpu().detach().numpy()

# Sampled Dynamics
box_side = 0.03
start_point = x_train[1005, :]
# start_point = x_train[0, :]
# start_point = torch.tensor([-0.5, -0.66])
a = [start_point[0] - box_side, start_point[1] - box_side]
b = [start_point[0] + box_side, start_point[1] + box_side]

T = 20
dt = 0.01
num_samples = 3
init_state = torch.cat((torch.FloatTensor(num_samples, 1).uniform_(
    a[0], b[0]), torch.FloatTensor(num_samples, 1).uniform_(a[1], b[1]), torch.zeros((num_samples, dim))), dim=1)
samples = ds.integrate(init_state, T, dt)

# Vector Field
if "first" in order:
    field = ds(X_test[:, :dim])
elif "second" in order:
    field = dt * ds(X_test)

x_field = field[:, 0].reshape(resolution, -1).detach().numpy()
y_field = field[:, 1].reshape(resolution, -1).detach().numpy()

# Plot potential contour
fig = plt.figure()
ax = fig.add_subplot(111)
fig.colorbar(mappable,  ax=ax, label=r"$\phi$")

ax.contourf(x_m, y_m, phi, 500, cmap="jet")
ax.contour(x_m, y_m, phi, 10, cmap=None, colors='#f2e68f')
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

ax.streamplot(x_m.detach().numpy(), y_m.detach().numpy(), x_field,
              y_field, color=phi, cmap="jet")
for i in range(num_samples):
    ax.plot(samples.detach().numpy()[:, i, 0],
            samples.detach().numpy()[:, i, 1], color='k')
rect = patches.Rectangle((start_point[0] - box_side, start_point[1] - box_side),
                         2*box_side, 2*box_side, linewidth=1, edgecolor='k', facecolor='none')
ax.add_patch(rect)
ax.scatter(x_train[::10, 0], x_train[::10, 1],
           s=20, edgecolors='k', c='red')
ax.scatter(x_train[-1, 0], x_train[-1, 1], s=100,
           edgecolors='k', c='yellow', marker="*")
if obstacle:
    # x_obs = x_obs.cpu().detach().numpy()
    # circ = plt.Circle((x_obs[0, 0], x_obs[0, 1]), r_obs,
    #                   color='k', fill='grey', alpha=0.5)
    # ax.add_patch(circ)
    x_obs = x_obs.cpu().detach().numpy()
    ax.scatter(x_obs[:, 0], x_obs[:, 1])

ax.axis('square')
ax.set_xlabel('$x^1$')
ax.set_ylabel('$x^2$')

# Metric
xy_m = torch.cat((torch.ravel(x_m).unsqueeze(-1),
                 torch.ravel(y_m).unsqueeze(-1)), dim=1).requires_grad_(True)
Z = ds.embedding(xy_m)
dZ = ds.embedding.jacobian(xy_m, Z)
M = ds.embedding.pullmetric(Z, dZ)
detM = M.det()

num_elps = 10
x_e, y_e = torch.meshgrid(torch.linspace(grid[0], grid[1], steps=num_elps),
                          torch.linspace(grid[2], grid[3], steps=num_elps))
xy_e = torch.cat((torch.ravel(x_e).unsqueeze(-1),
                 torch.ravel(y_e).unsqueeze(-1)), dim=1).requires_grad_(True)
z_m = ds.embedding(xy_e)
dz_m = ds.embedding.jacobian(xy_e, z_m)
L, V = torch.linalg.eig(ds.embedding.pullmetric(z_m, dz_m))
L = torch.real(L)
V = torch.real(V)

theta = torch.linspace(0, 2*torch.pi, steps=50)
ellipses = torch.cat(((L[:, 0].unsqueeze(-1)*theta.cos()).unsqueeze(-1),
                     (L[:, 1].unsqueeze(-1)*theta.sin()).unsqueeze(-1)), dim=2)
ellipses = torch.bmm(V, ellipses.permute(0, 2, 1)).permute(0, 2, 1)

detM = detM.reshape(resolution, -1).detach().numpy()
detM -= np.min(detM)
detM /= np.max(detM)
colors = plt.cm.jet(detM)
mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
mappable.set_array(detM)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(x_m.detach().numpy(), y_m.detach().numpy(), detM, 500, cmap="jet")
fig.colorbar(mappable,  ax=ax, label=r"$det(g)$")
ax.scatter(x_e, y_e, color="k", s=0.1)
xy_e = xy_e.detach().numpy()
# ellipses = 1e-3*ellipses.detach().numpy()
# for i in range(1):
#     ax.plot(xy_e[i, 0] + ellipses[i, :, 0],
#             xy_e[i, 1] + ellipses[i, :, 1], color="k", linewidth=0.5)
# ax.contour(x, y, detM, 10, cmap=None, colors='#f2e68f')
ax.axis("equal")

plt.show()
