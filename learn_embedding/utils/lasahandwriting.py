#!/usr/bin/env python

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from .data_process import DataProcess


class LasaHandwriting():
    def __init__(self, path=None):
        if path is not None:
            self._path = path

    def load(self):
        data = sio.loadmat(self.path)

        self.dt = data["dt"]
        self.time = [demo['t'][0, 0].T for demo in data['demos'][0]]
        self.pos = [demo['pos'][0, 0].T for demo in data['demos'][0]]
        self.vel = [demo['vel'][0, 0].T for demo in data['demos'][0]]
        self.acc = [demo['acc'][0, 0].T for demo in data['demos'][0]]

        return self

    def process(self):
        for i in range(len(self.time)):
            # Trim
            self.time[i] = DataProcess.trim(self.time[i], 10, 10)
            self.pos[i] = DataProcess.trim(self.pos[i], 10, 10)
            # self.vel[i]  = DataProcess.trim(self.vel[i], 10, 10)
            # self.acc[i]  = DataProcess.trim(self.acc[i], 10, 10)

            # Smooth
            self.pos[i] = DataProcess.smooth(self.pos[i], 10, 2, deriv=0, delta=self.dt, mode='interp', cval=0.0)
            self.pos[i] -= self.pos[i][-1, :]
            self.vel[i] = DataProcess.smooth(self.pos[i], 10, 2, deriv=1, delta=self.dt, mode='interp', cval=0.0)
            self.acc[i] = DataProcess.smooth(self.pos[i], 10, 2, deriv=2, delta=self.dt, mode='interp', cval=0.0)

        return self

    def dataset(self, target="acceleration", split=0.6, visualize=False):
        split = int(np.round(len(self.time)*split))

        if target == "acceleration":
            train_x = np.concatenate((self.pos[0], self.vel[0]), axis=1)
            train_y = self.acc[0]
            for i in range(1, split):
                train_x = np.append(train_x, np.concatenate((self.pos[i], self.vel[i]), axis=1), axis=0)
                train_y = np.append(train_y, self.acc[i], axis=0)

            test_x = np.concatenate((self.pos[split], self.vel[split]), axis=1)
            test_y = self.acc[split]
            for i in range(split+1, len(self.time)):
                test_x = np.append(test_x, np.concatenate((self.pos[i], self.vel[i]), axis=1), axis=0)
                test_y = np.append(test_y, self.acc[i], axis=0)

            if visualize:
                fig = plt.figure(figsize=(15, 5))
                ax = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
                ax[0].scatter(train_x[:, 0], train_x[:, 1], s=20, c='r')
                ax[1].scatter(train_x[:, 2], train_x[:, 3], s=20, c='g')
                ax[2].scatter(train_y[:, 0], train_y[:, 1], s=20, c='b')
                ax[0].axis('equal')
                ax[1].axis('equal')
                ax[2].axis('equal')
                fig = plt.figure(figsize=(15, 5))
                ax = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
                ax[0].scatter(test_x[:, 0], test_x[:, 1], s=20, c='r')
                ax[1].scatter(test_x[:, 2], test_x[:, 3], s=20, c='g')
                ax[2].scatter(test_y[:, 0], test_y[:, 1], s=20, c='b')
                ax[0].axis('equal')
                ax[1].axis('equal')
                ax[2].axis('equal')
        else:
            train_x = self.pos[0]
            train_y = self.vel[0]
            for i in range(1, split):
                train_x = np.append(train_x, self.pos[i], axis=0)
                train_y = np.append(train_y, self.vel[i], axis=0)

            test_x = self.pos[split]
            test_y = self.vel[split]
            for i in range(split+1, len(self.time)):
                test_x = np.append(test_x, self.pos[i], axis=0)
                test_y = np.append(test_y, self.vel[i], axis=0)

            if visualize:
                fig = plt.figure(figsize=(10, 5))
                ax = [fig.add_subplot(121), fig.add_subplot(122)]
                ax[0].scatter(train_x[:, 0], train_x[:, 1], s=20, c='r')
                ax[1].scatter(train_y[:, 0], train_y[:, 1], s=20, c='g')
                ax[0].axis('equal')
                ax[1].axis('equal')
                fig = plt.figure(figsize=(10, 5))
                ax = [fig.add_subplot(121), fig.add_subplot(122)]
                ax[0].scatter(test_x[:, 0], test_x[:, 1], s=20, c='r')
                ax[1].scatter(test_y[:, 0], test_y[:, 1], s=20, c='g')
                ax[0].axis('equal')
                ax[1].axis('equal')

        return train_x, train_y, test_x, test_y

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value
