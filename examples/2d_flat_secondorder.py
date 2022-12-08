#!/usr/bin/env python

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from learn_embedding.utils import create_model, squared_exp, infty_exp, linear_map

# User input
obstacle = sys.argv[1] if len(sys.argv) > 1 else None

# CPU/GPU setting
use_cuda = False  # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Test data
dim = 2
resolution = 100
y, x = torch.meshgrid(torch.linspace(-1, 1, resolution),
                      torch.linspace(-1, 1, resolution))
X = torch.cat((torch.ravel(x).unsqueeze(-1), torch.ravel(y).unsqueeze(-1),
              torch.zeros((resolution**2, dim))), dim=1).requires_grad_(True)

# Model
ds = create_model(torch.zeros((1, 4)), "second_flat")

# Obstacle
if obstacle is not None:
    # ds.embedding.metric = metric_exp
    # ds.embedding.metric = metric_infty

    if obstacle == "convex":
        r_obs = 0.1
        x_obs = torch.tensor([[-0.4,   -0.4]]).to(device)

        # ds.embedding.deformation = lambda x, y: infty_exp(
        #     x, y, r=0.005, a=0.005, b=2)
        ds.embedding.deformation = lambda x, y: squared_exp(
            x, y, sigma=0.1, eta=10)
    elif obstacle == "concave":
        barrier = torch.linspace(-np.pi/5, 0.8*np.pi,
                                 50).unsqueeze(1).to(device)
        c = torch.tensor([[-0.4,   -0.4]]).to(device)
        r_obs = 0.05
        x_obs = c + 0.2*torch.cat((np.cos(barrier), np.sin(barrier)), axis=1)

        # ds.embedding.deformation = lambda x, y: infty_exp(
        #     x, y, r=0.005, a=0.005, b=2)
        ds.embedding.deformation = lambda x, y: squared_exp(
            x, y, sigma=0.03, eta=10)

    ds.embedding.obstacles = x_obs
    y_obs = ds.embedding(x_obs)

# Potential
potential = ds.potential(X[:, :dim]).reshape(resolution, -1)

# Embedding
embedding = ds.embedding(X[:, :dim]).reshape(resolution, -1, dim+1)

# Sampled Dynamics
box_side = 0.03
start_point = torch.tensor([-0.9, -0.9])
a = [start_point[0] - box_side, start_point[1] - box_side]
b = [start_point[0] + box_side, start_point[1] + box_side]

T = 3
dt = 0.005
num_samples = 1
init_state = torch.tensor([[-0.9, -0.85, 1, 1]])
# init_state = torch.cat((torch.FloatTensor(num_samples, 1).uniform_(
#     a[0], b[0]), torch.FloatTensor(num_samples, 1).uniform_(a[1], b[1]), torch.ones((num_samples, dim))), dim=1)
samples = ds.integrate_geodesic(init_state, T, dt)
samples2 = ds.integrate(init_state, T + 5, dt)

# Vector Field
field = dt * ds(X).reshape(resolution, -1, dim)

# Metric
pos = X[:, :dim]
Z = ds.embedding(pos)
dZ = ds.embedding.jacobian(pos, Z)
M = ds.embedding.pullmetric(Z, dZ)
detM = M.det().reshape(resolution, -1)

num_elps = 10
y_coarse, x_coarse = torch.meshgrid(
    torch.linspace(-0.8, 0.8, steps=num_elps), torch.linspace(-0.8, 0.8, steps=num_elps))
X_coarse = torch.cat((torch.ravel(x_coarse).unsqueeze(-1),
                      torch.ravel(y_coarse).unsqueeze(-1)), dim=1).requires_grad_(True)
Z_coarse = ds.embedding(X_coarse)
dZ_coarse = ds.embedding.jacobian(X_coarse, Z_coarse)
L_coarse, V_coarse = torch.linalg.eig(
    ds.embedding.pullmetric(Z_coarse, dZ_coarse))
L_coarse = torch.real(L_coarse)
V_coarse = torch.real(V_coarse)

step = 40
X_samples = samples[::step, 0, :dim]
Vel_samples = samples[::step, 0, dim:]
Z_samples = ds.embedding(X_samples)
dZ_samples = ds.embedding.jacobian(X_samples, Z_samples)
M_samples = ds.embedding.pullmetric(Z_samples, dZ_samples)
L_samples, V_samples = torch.linalg.eig(M_samples)
L_samples = torch.real(L_samples)
V_samples = torch.real(V_samples)


theta = torch.linspace(0, 2*torch.pi, steps=50)

ellipses_coarse = torch.cat(((L_coarse[:, 0].unsqueeze(-1)*theta.cos()).unsqueeze(-1),
                             (L_coarse[:, 1].unsqueeze(-1)*theta.sin()).unsqueeze(-1)), dim=2)
ellipses_coarse = torch.bmm(
    V_coarse, ellipses_coarse.permute(0, 2, 1)).permute(0, 2, 1)

for i in range(ellipses_coarse.shape[0]):
    ellipses_coarse[i, :, 0] = linear_map(ellipses_coarse[i, :, 0], ellipses_coarse[i, :, 0].min(
    ), ellipses_coarse[i, :, 0].max(), -0.05, 0.05)
    ellipses_coarse[i, :, 1] = linear_map(ellipses_coarse[i, :, 1], ellipses_coarse[i, :, 1].min(
    ), ellipses_coarse[i, :, 1].max(), -0.05, 0.05)

ellipses_samples = torch.cat(((L_samples[:, 0].unsqueeze(-1)*theta.cos()).unsqueeze(-1),
                             (L_samples[:, 1].unsqueeze(-1)*theta.sin()).unsqueeze(-1)), dim=2)
ellipses_samples = torch.bmm(
    V_samples, ellipses_samples.permute(0, 2, 1)).permute(0, 2, 1)

for i in range(ellipses_samples.shape[0]):
    ellipses_samples[i, :, 0] = linear_map(ellipses_samples[i, :, 0], ellipses_samples[i, :, 0].min(
    ), ellipses_samples[i, :, 0].max(), -0.05, 0.05)
    ellipses_samples[i, :, 1] = linear_map(ellipses_samples[i, :, 1], ellipses_samples[i, :, 1].min(
    ), ellipses_samples[i, :, 1].max(), -0.05, 0.05)


# Christoffel symbols
C_samples = ds.embedding.christoffel(X_samples, M_samples)
Cv_samples = torch.bmm(M_samples,torch.einsum('bqij,bi->bqj', C_samples, Vel_samples))
# Cv_samples = ds.embedding.coriolis(X_samples, Vel_samples, M_samples)
L_christoffel, V_christoffel = torch.linalg.eig(Cv_samples)
L_christoffel = torch.real(L_christoffel)
V_christoffel = torch.real(V_christoffel)

ellipses_christoffel = torch.cat(((L_christoffel[:, 0].unsqueeze(-1)*theta.cos()).unsqueeze(-1),
                                  (L_christoffel[:, 1].unsqueeze(-1)*theta.sin()).unsqueeze(-1)), dim=2)
ellipses_christoffel = torch.bmm(
    V_christoffel, ellipses_christoffel.permute(0, 2, 1)).permute(0, 2, 1)

for i in range(ellipses_christoffel.shape[0]):
    ellipses_christoffel[i, :, 0] = linear_map(ellipses_christoffel[i, :, 0], ellipses_christoffel[i, :, 0].min(
    ), ellipses_christoffel[i, :, 0].max(), -0.05, 0.05)
    ellipses_christoffel[i, :, 1] = linear_map(ellipses_christoffel[i, :, 1], ellipses_christoffel[i, :, 1].min(
    ), ellipses_christoffel[i, :, 1].max(), -0.05, 0.05)


# Plot potential contour
with torch.no_grad():
    fig = plt.figure()

    # Plot potential
    colors = plt.cm.jet(potential)
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    mappable.set_array(potential)

    ax = fig.add_subplot(231)
    fig.colorbar(mappable,  ax=ax, label=r"$\phi$")

    ax.contourf(x, y, potential, 500, cmap="jet")
    ax.contour(x, y, potential, 10, cmap=None, colors='#f2e68f')

    ax.axis('square')
    ax.set_xlabel('$x^1$')
    ax.set_ylabel('$x^2$')
    ax.set_title('Potential Function $\phi$')

    # Plot embedding
    ax = fig.add_subplot(232, projection="3d")
    fig.colorbar(mappable,  ax=ax, label=r"$\phi$")

    ax.plot_surface(embedding[:, :, 0], embedding[:, :, 1], embedding[:, :, 2],
                    facecolors=colors, antialiased=True, linewidth=0, alpha=0.5)
    # if obstacle:
    #     y_obs = y_obs.cpu().detach().numpy()
    #     u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    #     obs_x = y_obs[0, 0] + r_obs*np.cos(u)*np.sin(v)
    #     obs_y = y_obs[0, 1] + r_obs*np.sin(u)*np.sin(v)
    #     obs_z = y_obs[0, 2] + 0.6*np.cos(v)
    #     ax.plot_surface(obs_x, obs_y, obs_z, linewidth=0.0,
    #                     cstride=1, rstride=1)

    ax.set_xlabel('$y^1$')
    ax.set_ylabel('$y^2$')
    ax.set_zlabel('$y^3$')
    ax.set_title('Embedding Space')

    # Plot vector field
    ax = fig.add_subplot(233)
    fig.colorbar(mappable,  ax=ax, label=r"$\phi$")
    for i in range(num_samples):
        ax.plot(samples[:, i, 0],samples[:, i, 1], color='k')
        ax.plot(samples2[:, i, 0],samples2[:, i, 1], color='r')
    ax.streamplot(x.numpy(), y.numpy(), field[:, :, 0], field[:, :, 1],
                  color=potential.numpy(), cmap="jet")
    if obstacle is not None:
        for i in range(x_obs.shape[0]):
            circ = plt.Circle((x_obs[i, 0], x_obs[i, 1]), r_obs,
                              color='k', fill='grey', alpha=0.5)
            ax.add_patch(circ)
    ax.axis('square')
    ax.set_xlabel('$x^1$')
    ax.set_ylabel('$x^2$')
    ax.set_title('Vector Field')

    # Metric Determinant
    ax = fig.add_subplot(234)
    colors = plt.cm.jet(detM)
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    mappable.set_array(detM)

    ax.contourf(x, y, detM, 500, cmap="jet")
    fig.colorbar(mappable,  ax=ax, label=r"$det(g)$")
    ax.scatter(x_coarse, y_coarse, color="k", s=0.1)
    for i in range(ellipses_coarse.shape[0]):
        ax.plot(X_coarse[i, 0] + ellipses_coarse[i, :, 0],
                X_coarse[i, 1] + ellipses_coarse[i, :, 1], color="k", linewidth=0.5)
    ax.axis('square')
    ax.set_title('Metric Determinant')

    # Sampled trajectory + metric ellipses
    ax = fig.add_subplot(235)
    fig.colorbar(mappable,  ax=ax, label=r"$\phi$")
    for i in range(num_samples):
        ax.plot(samples[:, i, 0],samples[:, i, 1], color='k')
        ax.plot(samples2[:, i, 0],samples2[:, i, 1], color='r')
    ax.scatter(X_samples[:, 0], X_samples[:, 1], color="k", s=10)
    for i in range(ellipses_samples.shape[0]):
        ax.plot(X_samples[i, 0] + ellipses_samples[i, :, 0],
                X_samples[i, 1] + ellipses_samples[i, :, 1], color="k", linewidth=0.5)

    # rect = patches.Rectangle((start_point[0] - box_side, start_point[1] - box_side),
    #                          2*box_side, 2*box_side, linewidth=1, edgecolor='k', facecolor='none')
    # ax.add_patch(rect)
    ax.axis('square')
    ax.set_xlabel('$x^1$')
    ax.set_ylabel('$x^2$')
    ax.set_title('Sampled Trajectory with Metric Ellipses')

    # Sampled trajectory + christoffel ellipses
    ax = fig.add_subplot(236)
    fig.colorbar(mappable,  ax=ax, label=r"$\phi$")
    for i in range(num_samples):
        ax.plot(samples[:, i, 0],samples[:, i, 1], color='k')
        ax.plot(samples2[:, i, 0],samples2[:, i, 1], color='r')
    ax.scatter(X_samples[:, 0], X_samples[:, 1], color="k", s=10)
    for i in range(ellipses_christoffel.shape[0]):
        ax.plot(X_samples[i, 0] + ellipses_christoffel[i, :, 0],
                X_samples[i, 1] + ellipses_christoffel[i, :, 1], color="k", linewidth=0.5)

    # rect = patches.Rectangle((start_point[0] - box_side, start_point[1] - box_side),
    #                          2*box_side, 2*box_side, linewidth=1, edgecolor='k', facecolor='none')
    # ax.add_patch(rect)
    ax.axis('square')
    ax.set_xlabel('$x^1$')
    ax.set_ylabel('$x^2$')
    ax.set_title('Sampled Trajectory with Christoffel Ellipses')

plt.show()
