import pickle
import numpy as np
import matplotlib.pyplot as plt
from .data_process import DataProcess


class RoboticDemos():
    def __init__(self, path=None):
        if path is not None:
            self._path = path

    def load(self):
        with open(self._path, 'rb') as fp:
            data = pickle.load(fp)

        self.pos = [demo['ee_pose'][:, :3] for demo in data]
        self.vel = [demo['ee_velocity'][:, :3] for demo in data]
        self.acc = []
        self.dt = data[0]['dt']

        return self

    def process(self):
        for i in range(len(self.pos)):
            # Trim
            self.pos[i] = DataProcess.trim(self.pos[i], 50, 50)
            self.vel[i] = DataProcess.trim(self.vel[i], 50, 50)

            # acc
            acc = (self.vel[i][1:, :] - self.vel[i][:-1, :])/self.dt
            acc = np.append(acc, np.zeros([1, self.pos[i].shape[1]]), axis=0)
            self.acc.append(acc)

            # Smooth
            # self.pos[i] = DataProcess.smooth(self.pos[i], 10, 2, deriv=0)
            # self.pos[i] -= self.pos[i][-1, :]
            # self.vel[i] = DataProcess.smooth(self.vel[i], 10, 2, deriv=0)
            # self.acc[i] = DataProcess.smooth(self.acc[i], 10, 2, deriv=0)
            self.pos[i] = DataProcess.smooth(self.pos[i], 10, 2, deriv=0, delta=self.dt, mode='interp', cval=0.0)
            self.pos[i] -= self.pos[i][-1, :]
            self.vel[i] = DataProcess.smooth(self.pos[i], 10, 2, deriv=1, delta=self.dt, mode='interp', cval=0.0)
            self.acc[i] = DataProcess.smooth(self.pos[i], 10, 2, deriv=2, delta=self.dt, mode='interp', cval=0.0)

        return self

    def dataset(self, target="acceleration", split=0.6, visualize=False):
        split = int(np.round(len(self.pos)*split))

        if target == "acceleration":
            train_x = np.concatenate((self.pos[0], self.vel[0]), axis=1)
            train_y = self.acc[0]
            for i in range(1, split):
                train_x = np.append(train_x, np.concatenate((self.pos[i], self.vel[i]), axis=1), axis=0)
                train_y = np.append(train_y, self.acc[i], axis=0)

            test_x = np.concatenate((self.pos[split], self.vel[split]), axis=1)
            test_y = self.acc[split]
            for i in range(split+1, len(self.pos)):
                test_x = np.append(test_x, np.concatenate((self.pos[i], self.vel[i]), axis=1), axis=0)
                test_y = np.append(test_y, self.acc[i], axis=0)

            if visualize:
                fig = plt.figure(figsize=(15, 5))
                ax = [fig.add_subplot(131, projection="3d"), fig.add_subplot(132, projection="3d"), fig.add_subplot(133, projection="3d")]
                ax[0].scatter(train_x[:, 0], train_x[:, 1], train_x[:, 2], c='r')
                ax[1].scatter(train_x[:, 3], train_x[:, 4], train_x[:, 5], c='g')
                ax[2].scatter(train_y[:, 0], train_y[:, 1], train_y[:, 2], c='b')
                ax[0].axis('equal')
                ax[1].axis('equal')
                ax[2].axis('equal')
                fig = plt.figure(figsize=(15, 5))
                ax = [fig.add_subplot(131, projection="3d"), fig.add_subplot(132, projection="3d"), fig.add_subplot(133, projection="3d")]
                ax[0].scatter(test_x[:, 0], test_x[:, 1], test_x[:, 2], c='r')
                ax[1].scatter(test_x[:, 2], test_x[:, 3], test_x[:, 5], c='g')
                ax[2].scatter(test_y[:, 0], test_y[:, 1], test_y[:, 2], c='b')
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
            for i in range(split+1, len(self.pos)):
                test_x = np.append(test_x, self.pos[i], axis=0)
                test_y = np.append(test_y, self.vel[i], axis=0)

            if visualize:
                fig = plt.figure(figsize=(10, 5))
                ax = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]
                ax[0].scatter(train_x[:, 0], train_x[:, 1], train_x[:, 2], c='r')
                ax[1].scatter(train_y[:, 0], train_y[:, 1], train_y[:, 2], c='g')
                ax[0].axis('equal')
                ax[1].axis('equal')
                fig = plt.figure(figsize=(10, 5))
                ax = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]
                ax[0].scatter(test_x[:, 0], test_x[:, 1], test_x[:, 2], c='r')
                ax[1].scatter(test_y[:, 0], test_y[:, 1], test_y[:, 2], c='g')
                ax[0].axis('equal')
                ax[1].axis('equal')

        return train_x, train_y, test_x, test_y

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value
