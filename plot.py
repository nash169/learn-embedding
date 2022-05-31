#!/usr/bin/env python

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.dynamics import Dynamics

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data
dataset = "BendedLine"
data = np.loadtxt(os.path.join('data', '{}.csv'.format(dataset)))

# Tensor data
X = torch.from_numpy(data[:, :4]).float().to(device).requires_grad_(True)
Y = torch.from_numpy(data[:, 4:6]).float().to(device).requires_grad_(True)

# Plot data
x_train = data[:, :2]
dx_train = data[:, 2:4]
ddx_train = data[:, 4:6]

# Test data
resolution = 100
x_mesh, y_mesh = np.meshgrid(np.linspace(-60, 60, resolution),
                             np.linspace(-60, 60, resolution))
x_test = np.array(
    [x_mesh.ravel(order="F"), y_mesh.ravel(order="F")]).transpose()
X_test = np.zeros([x_test.shape[0], 2*x_test.shape[1]])
X_test[:, :x_test.shape[1]] = x_test

x_test = torch.from_numpy(
    x_test).float().to(device).requires_grad_(True)
X_test = torch.from_numpy(
    X_test).float().to(device).requires_grad_(True)


# Net model
dim = 2
structure = [100, 100, 100]
attractor = torch.tensor([0, 0]).to(device)
K = torch.eye(2, 2).to(device)
D = torch.eye(2, 2).to(device)

ds = Dynamics(2, attractor, structure).to(device)
ds.dissipation = (D, False)
ds.stiffness = (K, False)

ds.load_state_dict(torch.load(os.path.join(
    'models', '{}.pt'.format(dataset)), map_location=torch.device(device)))

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
embedding = ds.embedding(x_test)
embedding = embedding.cpu().detach().numpy()
x_embedding = embedding[:, 0].reshape(resolution, -1, order="F")
y_embedding = embedding[:, 1].reshape(resolution, -1, order="F")
z_embedding = embedding[:, 2].reshape(resolution, -1, order="F")
train_embedding = ds.embedding(X[:, :2])
train_embedding = train_embedding.cpu().detach().numpy()

# Sampled Dynamics
box_side = 10
a = [x_train[0, 0] - box_side, x_train[0, 1] - box_side]
b = [x_train[0, 0] + box_side, x_train[0, 1] + box_side]

T = 1
dt = 0.01
steps = int(np.ceil(T/dt))
num_samples = 3
samples = []

for i in range(num_samples):
    state = np.zeros([steps, 2*dim])
    state[0, 0] = np.random.uniform(a[0], b[0])
    state[0, 1] = np.random.uniform(a[1], b[1])
    samples.append(state)

for step in range(steps-1):
    for i in range(num_samples):
        X_sample = torch.from_numpy(samples[i][step, :]).float().to(
            device).requires_grad_(True).unsqueeze(0)
        samples[i][step+1, dim:] = samples[i][step, dim:] + \
            dt*ds(X_sample).cpu().detach().numpy()
        samples[i][step+1, :dim] = samples[i][step, :dim] + \
            dt*samples[i][step+1, dim:]

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
ax.scatter(x_train[::10, 0], x_train[::10, 1], s=20, edgecolors='k', c='red')
ax.axis('square')

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x_embedding, y_embedding, z_embedding,
                facecolors=colors, antialiased=True, linewidth=0, alpha=0.5)
fig.colorbar(mappable,  ax=ax, label=r"$\phi$")
ax.set_box_aspect((np.ptp(x_embedding), np.ptp(
    y_embedding), np.ptp(z_embedding)))
ax.scatter(train_embedding[::10, 0], train_embedding[::10, 1],
           train_embedding[::10, 2], s=20, edgecolors='k', c='red')

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

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(x_train[::10, 0], x_train[::10, 1], s=20, edgecolors='k', c='red')
# ax.quiver()

plt.show()
