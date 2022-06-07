#!/usr/bin/env python

import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import sys

from src.utils import linear_map

dataset = sys.argv[1] if len(sys.argv) > 1 else "Angle"
num_trajs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
init_cut = 50

data = sio.loadmat(os.path.join('data', '{}.mat'.format(dataset)))
data = data['demos']

dim = 2
lower, upper = -0.5, 0.5
rescale, vel, acc = True, True, True

pos = data[0, 0]['pos'][0, 0].T
for i in range(1, num_trajs):
    pos = np.append(pos, data[0, i]['pos'][0, 0].T, axis=0)
x_min, x_max = np.min(pos[:, 0]), np.max(pos[:, 0])
y_min, y_max = np.min(pos[:, 1]), np.max(pos[:, 1])

trajs = []
for i in range(num_trajs):
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


trainset = trajs[0][init_cut:, 1:]
for i in range(1, num_trajs):
    trainset = np.append(trainset, trajs[i][init_cut:, 1:], axis=0)

np.savetxt('data/' + dataset + '.csv', trainset)

fig = plt.figure()
ax = fig.add_subplot(111)
colors = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf",
          "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00", ]
for i in range(num_trajs):
    ax.scatter(trajs[i][init_cut::10, 1], trajs[i][init_cut::10, 2],
               s=20, edgecolors='k', c=colors[i])
    ax.quiver(trajs[i][init_cut::10, 1], trajs[i][init_cut::10, 2],
              trajs[i][init_cut::10, 3], trajs[i][init_cut::10, 4], scale=100, width=0.002, color='r')
    ax.quiver(trajs[i][init_cut::10, 1], trajs[i][init_cut::10, 2],
              trajs[i][init_cut::10, 5], trajs[i][init_cut::10, 6], scale=100, width=0.002, color='b')

plt.show()
