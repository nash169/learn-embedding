#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P


class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)


class MatrixExponential(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)


class Spherical(nn.Module):
    def __init__(self, in_features):
        super(Spherical, self).__init__()

        self.spherical = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return nn.functional.linear(x, self.spherical.abs()*torch.eye(x.shape[1]).to(x.device))


class Diagonal(nn.Module):
    def __init__(self, in_features):

        super(Diagonal, self).__init__()

        self.diagonal = nn.Linear(in_features, in_features, bias=False)

    def forward(self, x):
        self.diagonal.weight.data *= torch.eye(
            x.shape[1], dtype=bool).to(x.device)
        self.diagonal.weight.data = torch.abs(self.diagonal.weight.data)

        return self.diagonal(x)


class SPD(nn.Module):
    def __init__(self, in_features):

        super(SPD, self).__init__()

        self.spd = nn.Linear(in_features, in_features, bias=False)
        P.register_parametrization(self.spd, "weight", Symmetric())
        P.register_parametrization(self.spd, "weight", MatrixExponential())

    def forward(self, x):
        return self.spd(x)
