#!/usr/bin/env python

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy

from src.kernel_machine import KernelMachine
from src.coupling_layer import CouplingLayer
from src.feedforward import FeedForward
from src.embedding import Embedding
from src.parametrization import SPD, Diagonal, Spherical, Fixed
from src.dynamics_second import DynamicsSecond
from src.dynamics_first import DynamicsFirst
from src.trainer import Trainer

from fastdtw import fastdtw
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Datasets
l = os.listdir('trainset')
li = [x.split('.')[0] for x in l]
li = ['Angle', 'Khamesh']

# Number of reps
reps = 2

for dataset in li:
    dtw = []
    cos = []
    rmse = []
    for n in range(reps):
        # Data
        data = np.loadtxt(os.path.join('trainset', '{}.csv'.format(dataset)))
        dim = int(data.shape[1]/3)
        x_train = data[:, :dim]

        # State (pos,vel)
        X = torch.from_numpy(data[:, :2*dim]).float().to(device)
        Y = torch.from_numpy(data[:, 2*dim:]).float().to(device)
        attractor = X[-1, :dim]

        # Function Approximator
        approximator = FeedForward(dim, [64], 1)

        # Model 1
        embedding_1 = Embedding(approximator)
        stiffness_1 = SPD(dim)
        ds1 = DynamicsFirst(attractor, stiffness_1, embedding_1).to(device)

        # Trainer 1
        trainer = Trainer(ds1, X[:, :dim], X[:, dim:])
        trainer.optimizer = torch.optim.Adam(
            trainer.model.parameters(), lr=1e-3, weight_decay=1e-8)
        # trainer.loss = torch.nn.MSELoss()
        trainer.loss = torch.nn.SmoothL1Loss()
        trainer.options(normalize=False, shuffle=True, print_loss=True,
                        epochs=10000, load_model=None)
        trainer.train()
        trainer.save(dataset+"1")

        # Model 2
        embedding_2 = embedding_1
        stiffness_2 = stiffness_1
        dissipation = copy.deepcopy(stiffness_1)
        ds = DynamicsSecond(attractor, stiffness_2,
                            dissipation, embedding_2).to(device)
        ds.velocity_ = ds1

        # # Fix first DS parameters
        # for param in ds1.parameters():
        #     param.requires_grad = False

        # Trainer 2
        trainer.model = ds
        trainer.input = X
        trainer.target = Y
        trainer.optimizer = torch.optim.Adam(
            trainer.model.parameters(), lr=1e-3, weight_decay=1e-8)
        trainer.options(epochs=10000, load_model=None)
        trainer.train()
        trainer.save(dataset+"2")

        # Test data
        resolution = 100
        lower, upper = -0.5, 0.5
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

        # Vector Field
        field = X_test[:, dim:] + 0.01 * ds(X_test)
        field = field.cpu().detach().numpy()
        x_field = field[:, 0].reshape(resolution, -1, order="F")
        y_field = field[:, 1].reshape(resolution, -1, order="F")

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x_embedding, y_embedding, z_embedding,
                        facecolors=colors, antialiased=True, linewidth=0, alpha=0.5)
        fig.colorbar(mappable,  ax=ax, label=r"$\phi$")
        # ax.set_box_aspect((np.ptp(x_embedding), np.ptp(
        #     y_embedding), np.ptp(z_embedding)))
        ax.scatter(train_embedding[::10, 0], train_embedding[::10, 1],
                   train_embedding[::10, 2], s=20, edgecolors='k', c='red')
        ax.set_xlabel('$y^1$')
        ax.set_ylabel('$y^2$')
        ax.set_zlabel('$y^3$')
        fig.savefig('images/'+dataset+'_manifold.png', format='png')
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.streamplot(x_mesh, y_mesh, x_field, y_field, color=phi, cmap="jet")
        ax.axis('square')
        fig.colorbar(mappable,  ax=ax, label=r"$\phi$")
        ax.scatter(x_train[::10, 0], x_train[::10, 1],
                   s=20, edgecolors='k', c='red')
        ax.scatter(x_train[-1, 0], x_train[-1, 1], s=100,
                   edgecolors='k', c='yellow', marker="*")
        fig.savefig('images/'+dataset+'_field.png', format='png')
        plt.close(fig)

        # Testing
        data = np.loadtxt(os.path.join('testset', '{}.csv'.format(dataset)))
        X = torch.from_numpy(
            data[:, :2*dim]).float().to(device).requires_grad_(True)
        Y = torch.from_numpy(
            data[:, 2*dim:]).float().to(device).requires_grad_(True)

        field = ds1(X[:, :dim])
        field = field.cpu().detach().numpy()
        field_check = X[:, :dim].cpu().detach().numpy()

        dist, _ = fastdtw(field, field_check, dist=cosine)
        dtw.append(dist/1000/6)
        cos.append(np.mean(1-cosine_similarity(field, field_check))/6)
        rmse.append(np.mean(np.linalg.norm(field-field_check, axis=1))/6)

    np.savetxt(os.path.join('results', '{}.csv'.format(dataset)),
               [dtw, cos, rmse])

# Evaluation
eval = np.zeros([len(li), 3, 2])
for i, dataset in enumerate(li):
    data = np.loadtxt(os.path.join('results', '{}.csv'.format(dataset)))
    if len(data.shape) > 1:
        eval[i, :, 0] = data.mean(axis=1)
        eval[i, :, 1] = data.std(axis=1)
    else:
        eval[i, :, 0] = data
        eval[i, :, 1] = 0

with open(os.path.join('results', '{}.csv'.format("eval")), "ab") as f:
    for i in range(3):
        np.savetxt(f, eval[:, i, 0][np.newaxis, :])
        np.savetxt(f, eval[:, i, 1][np.newaxis, :])
        f.write(b"\n")

data = [eval[:, 0, 0], eval[:, 1, 0], eval[:, 2, 0]]
fig, ax = plt.subplots()
green_diamond = dict(markerfacecolor='g', marker='D')
ax.boxplot(data, flierprops=green_diamond)
plt.xticks([1, 2, 3], ['RMSE', 'DTWD', 'CS'])
fig.savefig('images/eval.png', format='png')
plt.close(fig)
