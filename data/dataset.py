#!/usr/bin/env python

import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import sys
# import torch

from src.utils import linear_map
# from src.parametrization import Rotation

dataset = sys.argv[1] if len(sys.argv) > 1 else "Angle"
num_trajs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
init_cut = 20

data = sio.loadmat(os.path.join('data', '{}.mat'.format(dataset)))
data = data['demos']

dim = 2
lower, upper = -0.5, 0.5
rescale, vel, acc = True, True, True

pos = data[0, 0]['pos'][0, 0].T
for i in range(1, data.shape[1]):
    pos = np.append(pos, data[0, i]['pos'][0, 0].T, axis=0)
x_min, x_max = np.min(pos[:, 0]), np.max(pos[:, 0])
y_min, y_max = np.min(pos[:, 1]), np.max(pos[:, 1])

trajs = []
for i in range(data.shape[1]):
    # time
    t = data[0, i]['t'][0, 0].T
    dt = (t[1:] - t[:-1])

    # position
    x = data[0, i]['pos'][0, 0].T
    if rescale:
        x[:, 0] = linear_map(x[:, 0], x_min, x_max, lower, upper)
        x[:, 1] = linear_map(x[:, 1], y_min, y_max, lower, upper)

    # velocity
    if vel:
        v = np.divide(x[1:, :] - x[:-1, :], dt)
        v = np.append(v, np.zeros([1, dim]), axis=0)
    else:
        v = data[0, i]['vel'][0, 0].T

    # acceleration
    if acc:
        a = np.divide(v[1:, :] - v[:-1, :], dt)
        a = np.append(a, np.zeros([1, dim]), axis=0)
    else:
        a = data[0, i]['acc'][0, 0].T

    trajs.append(np.concatenate((t, x, v, a), axis=1))

# Trainig set
trainset = trajs[0][init_cut:, 1:]
for i in range(1, num_trajs):
    trainset = np.append(trainset, trajs[i][init_cut:, 1:], axis=0)

# # Augment training set via rotation
# num_data = trainset.shape[0]
# rot = Rotation()
# reps = 10
# for i in np.linspace(0+2*np.pi/reps, 2*np.pi, reps):
#     rot.rotation = torch.tensor(i, dtype=torch.float32)
#     pos = rot(torch.from_numpy(
#         trainset[:num_data, :dim]-trainset[num_data-1, :dim]).float()).cpu().detach().numpy() + trainset[num_data-1, :dim]
#     vel = rot(torch.from_numpy(
#         trainset[:num_data, dim:2*dim]).float()).cpu().detach().numpy()
#     acc = rot(torch.from_numpy(
#         trainset[:num_data, 2*dim:]).float()).cpu().detach().numpy()
#     trainset = np.append(trainset, np.concatenate(
#         (pos, vel, acc), axis=1), axis=0)

testset = trajs[num_trajs][init_cut:, 1:]
for i in range(num_trajs+1, len(trajs)):
    testset = np.append(testset, trajs[i][init_cut:, 1:], axis=0)

np.savetxt('data/train/' + dataset + '.csv', trainset)
np.savetxt('data/test/' + dataset + '.csv', testset)

# fig = plt.figure()
# ax = fig.add_subplot(111)
colors = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf",
          "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00", ]
# for i in range(num_trajs):
#     ax.scatter(trajs[i][init_cut::10, 1], trajs[i][init_cut::10, 2],
#                s=20, edgecolors='k', c=colors[i])
#     ax.quiver(trajs[i][init_cut::10, 1], trajs[i][init_cut::10, 2],
#               trajs[i][init_cut::10, 3], trajs[i][init_cut::10, 4], scale=100, width=0.002, color='r')
#     ax.quiver(trajs[i][init_cut::10, 1], trajs[i][init_cut::10, 2],
#               trajs[i][init_cut::10, 5], trajs[i][init_cut::10, 6], scale=100, width=0.002, color='b')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(trainset[init_cut::10, 0], trainset[init_cut::10, 1],
           s=20, edgecolors='k', c=colors[0])
ax.quiver(trainset[init_cut::10, 0], trainset[init_cut::10, 1],
          trainset[init_cut::10, 2], trainset[init_cut::10, 3], scale=100, width=0.002, color='r')
ax.quiver(trainset[init_cut::10, 0], trainset[init_cut::10, 1],
          trainset[init_cut::10, 4], trainset[init_cut::10, 5], scale=100, width=0.002, color='b')
ax.set_title("Training Set")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(testset[init_cut::10, 0], testset[init_cut::10, 1],
           s=20, edgecolors='k', c=colors[1])
ax.quiver(testset[init_cut::10, 0], testset[init_cut::10, 1],
          testset[init_cut::10, 2], testset[init_cut::10, 3], scale=100, width=0.002, color='r')
ax.quiver(testset[init_cut::10, 0], testset[init_cut::10, 1],
          testset[init_cut::10, 4], testset[init_cut::10, 5], scale=100, width=0.002, color='b')
ax.set_title("Testing Set")

plt.show()
