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
    def __init__(self):
        super(Spherical, self).__init__()

        self.spherical_ = nn.Parameter(torch.rand(1))

    def forward(self, x):
        # self.spherical_.clamp_min(0)
        # self.spherical_.data.clamp_min(0)
        return nn.functional.linear(x, self.spherical_.abs()*torch.eye(x.shape[1]).to(x.device))

    # Params
    @property
    def spherical(self):
        return self.spherical_

    @spherical.setter
    def spherical(self, value):
        self.spherical_ = nn.Parameter(value[0], requires_grad=value[1])


class Diagonal(nn.Module):
    def __init__(self, in_features):
        super(Diagonal, self).__init__()

        self.diagonal_ = nn.Parameter(torch.rand(in_features))
        # self.diagonal = nn.Linear(in_features, in_features, bias=False)

    def forward(self, x):
        return nn.functional.linear(x, torch.diag(self.diagonal_.abs()).to(x.device))
        # self.diagonal.weight.data *= torch.eye(
        #     x.shape[1], dtype=bool).to(x.device)
        # self.diagonal.weight.data = torch.abs(self.diagonal.weight.data)

        # return self.diagonal(x)

    # Params
    @property
    def diagonal(self):
        return self.diagonal_

    @diagonal.setter
    def diagonal(self, value):
        self.diagonal_ = nn.Parameter(value[0], requires_grad=value[1])


# class SPD(nn.Module):
#     def __init__(self, in_features):

#         super(SPD, self).__init__()

#         self.spd = nn.Linear(in_features, in_features, bias=False)
#         P.register_parametrization(self.spd, "weight", Symmetric())
#         P.register_parametrization(self.spd, "weight", MatrixExponential())

#     def forward(self, x):
#         return self.spd(x)


class Fixed(nn.Module):
    def __init__(self, in_features):

        super(Fixed, self).__init__()

        self.fixed_ = nn.Linear(in_features, in_features, bias=False)
        self.fixed_.weight = nn.Parameter(
            torch.eye(in_features, in_features), requires_grad=False)

    def forward(self, x):
        return self.fixed_(x)

     # Params
    @property
    def fixed(self):
        return self.fixed_

    @fixed.setter
    def fixed(self, value):
        self.fixed_.weight = nn.Parameter(value, requires_grad=False)


class SPD(nn.Module):
    def __init__(self, in_features):

        super(SPD, self).__init__()

        self.eig_ = nn.Parameter(torch.rand(in_features))
        self.vec_ = nn.Parameter(torch.rand(in_features))

    def forward(self, x):
        D = torch.diag(self.eig_).square()
        U, _ = torch.linalg.qr(torch.cat((self.vec_.unsqueeze(1),
                                          torch.rand(self.vec_.shape[0], self.vec_.shape[0]-1).to(x.device)), dim=1))
        return nn.functional.linear(x, torch.mm(U.transpose(1, 0), torch.mm(D, U)).to(x.device))
