#!/usr/bin/env python

import torch
import torch.nn as nn

from src.squared_exp import SquaredExp


class KernelExpansion(nn.Module):
    def __init__(self, samples, kernel=None):
        super(KernelExpansion, self).__init__()

        self.samples_ = samples

        self.weights_ = nn.Parameter(torch.rand(
            samples.shape[0]), requires_grad=True)

        if kernel is not None:
            self.kernel_ = kernel
        else:
            self.kernel_ = SquaredExp()

    def forward(self, x):
        return torch.mv(self.kernel(self.samples_, x).T, self.weights_).unsqueeze(1)

    # Sigma variance
    @property
    def kernel(self):
        return self.kernel_

    @kernel.setter
    def kernel(self, value):
        self.kernel_ = nn.Parameter(value[0], requires_grad=value[1])
