#!/usr/bin/env python

import numpy as np
from scipy.signal import savgol_filter


class DataProcess():
    # def __init__(self, time=None, position=None, velocity=None, acceleration=None):
    #     if time is not None:
    #         self._time = time

    #     if position is not None:
    #         self._position = position

    #     if velocity is not None:
    #         self._velocity = velocity

    #     if acceleration is not None:
    #         self._acceleration = acceleration

    # def generate_dataset(self, indices):
    #     pass

    @staticmethod
    def trim(x, lower, upper):
        p = x[lower:-upper - 1, :]
        return p

    @staticmethod
    def rescale(x, upper, lower):
        p = x - np.min(x, axis=0)
        p /= np.max(p, axis=0)
        p = lower + p*(upper-lower)
        return p

    @staticmethod
    def normalize(x):
        p = (x - np.mean(x, axis=0))/np.std(x, axis=0)
        return p

    @staticmethod
    def rotate(x, repetitions):
        p = np.zeros((x.shape[0], x.shape[1], repetitions))
        rotations = np.linspace(0+2*np.pi/repetitions, 2*np.pi, repetitions)
        for i, rot in enumerate(rotations):
            R = np.array([[np.cos(rot), -np.sin(rot)],
                          [np.sin(rot), np.cos(rot)]])
            p[:, :, i] = np.matmul(x, R)
        return p

    @staticmethod
    def derive(x, t):
        p = np.divide(x[1:, :] - x[:-1, :], t[1:] - t[:-1])
        p = np.append(p, np.zeros([1, x.shape[1]]), axis=0)
        return p

    @staticmethod
    def smooth(x, window_length, order=2, **kwargs):
        p = np.zeros_like(x)
        p = savgol_filter(x, window_length, order, axis=0, **kwargs)
        return p

    @staticmethod
    def subsample(x, num_samples):
        # Calculate total length of the trajectory
        total_length = np.sum(np.linalg.norm(x[1:] - x[:-1], axis=1))

        # Calculate average distance between consecutive samples
        average_distance = total_length / (num_samples - 1)

        # Add the first point to the subsampled trajectory
        idx = [0]

        # Iterate over the remaining points in the trajectory
        for j in range(1, x.shape[0]):
            # Calculate the distance to the previously added point
            distance = np.linalg.norm(x[j] - x[idx[-1]])

            # If the distance is greater than or equal to the average distance, add the point to the subsampled trajectory
            if distance >= average_distance:
                idx.append(j)

        # Add last point
        if x.shape[0] - 1 not in idx:
            idx = np.append(idx, x.shape[0] - 1)

        return np.array(idx)

    # @property
    # def time(self):
    #     return self._time

    # @time.setter
    # def time(self, value):
    #     self._time = value

    # @property
    # def position(self):
    #     return self._position

    # @position.setter
    # def position(self, value):
    #     self._position = value

    # @property
    # def velocity(self):
    #     return self._velocity

    # @velocity.setter
    # def velocity(self, value):
    #     self._velocity = value

    # @property
    # def acceleration(self):
    #     return self._acceleration

    # @acceleration.setter
    # def acceleration(self, value):
    #     self._acceleration = value
