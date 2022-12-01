#!/usr/bin/env python

import torch
import torch.nn as nn


class Diagonal(nn.Module):
    def __init__(self, in_features, grad=True):
        super(Diagonal, self).__init__()

        self.diagonal_ = nn.Parameter(
            torch.rand(in_features), requires_grad=grad)

    def forward(self, x):
        return nn.functional.linear(x, torch.diag(self.diagonal_.exp()).to(x.device))

    # Params
    @property
    def diagonal(self):
        return self.diagonal_

    @diagonal.setter
    def diagonal(self, value):
        self.diagonal_ = nn.Parameter(value[0], requires_grad=value[1])
