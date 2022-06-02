#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

from .parametrization import Symmetric, MatrixExponential


class Spherical(nn.Module):
    def __init__(self, in_features):
        super(Spherical, self).__init__()

        self.matrix = torch.eye(in_features, in_features)

        self.weights = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return nn.functional.linear(x, self.weights*self.matrix)


class Diagonal(nn.Module):
    def __init__(self, in_features):

        super(Diagonal, self).__init__()

        self.spd = nn.Linear(in_features, in_features, bias=False)

    def forward(self, x):
        self.spd.weight.data *= torch.eye(x.shape[1], dtype=bool).to(x.device)
        self.spd.weight.data = torch.abs(self.spd.weight.data)

        return self.spd(x)


class SPD(nn.Module):
    def __init__(self, in_features):

        super(SPD, self).__init__()

        self.spd = nn.Linear(in_features, in_features, bias=False)
        P.register_parametrization(self.spd, "weight", Symmetric())
        P.register_parametrization(self.spd, "weight", MatrixExponential())

    def forward(self, x):
        return self.spd(x)
