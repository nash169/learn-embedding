#!/usr/bin/env python

import os
from turtle import width
from matplotlib import colors, scale
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

dim = 2
dataset = "BendedLine"
data = sio.loadmat(os.path.join('data', '{}.mat'.format(dataset)))
data = data['demos']
num_trajs = data.shape[1]
trajs = []

for i in range(num_trajs):
    trajs.append(np.concatenate((data[0, i]['t'][0, 0].T,
                                 data[0, i]['pos'][0, 0].T,
                                 data[0, i]['vel'][0, 0].T,
                                 data[0, i]['acc'][0, 0].T
                                 ), axis=1))

    dt = (trajs[i][1:, 0] - trajs[i][:-1, 0])[:, np.newaxis]
    v = np.divide(trajs[i][1:, 1:dim+1] - trajs[i][:-1, 1:dim+1], dt)
    v = np.append(v, np.zeros([1, dim]), axis=0)
    trajs[i][:, dim+1:2*dim+1] = v

    a = np.divide(trajs[i][1:, dim+1:2*dim+1] -
                  trajs[i][:-1, dim+1:2*dim+1], dt)
    a = np.append(a, np.zeros([1, dim]), axis=0)
    trajs[i][:, -dim:] = a

select_trajs = 3
trainset = trajs[0][:, 1:]
for i in range(1, select_trajs):
    trainset = np.append(trainset, trajs[i][:, 1:], axis=0)

np.savetxt('data/' + dataset + '.csv', trainset)

colors = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf",
          "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00", ]

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(select_trajs):
    ax.scatter(trajs[i][::10, 1], trajs[i][::10, 2],
               s=20, edgecolors='k', c=colors[i])
    ax.quiver(trajs[i][::10, 1], trajs[i][::10, 2],
              trajs[i][::10, 3], trajs[i][::10, 4], scale=3000, width=0.002, color='r')
    ax.quiver(trajs[i][::10, 1], trajs[i][::10, 2],
              trajs[i][::10, 5], trajs[i][::10, 6], scale=3000, width=0.002, color='b')

plt.show()
