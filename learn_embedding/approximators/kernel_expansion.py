#!/usr/bin/env python

import torch
import torch.nn as nn

from ..kernels.squared_exp import SquaredExp


class KernelExpansion(nn.Module):
    def __init__(self, samples, kernel=None):
        super(KernelExpansion, self).__init__()

        self._samples = samples

        self._weights = nn.Parameter(0.1*torch.rand(samples.shape[0]), requires_grad=True)

        if kernel is not None:
            self._kernel = kernel
        else:
            self._kernel = SquaredExp()

    def forward(self, x):
        return torch.mv(self.kernel(self._samples, x).T, self.weights).unsqueeze(1)

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, value):
        self._kernel = value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = nn.Parameter(value)
