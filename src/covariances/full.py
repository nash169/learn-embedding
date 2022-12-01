#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P


class SPD(nn.Module):
    def __init__(self, in_features, grad=True):

        super(SPD, self).__init__()

        self.eigval_ = nn.Parameter(
            torch.rand(in_features), requires_grad=grad)
        self.eigvec_ = nn.Parameter(
            torch.rand(in_features), requires_grad=grad)

    def forward(self, x):
        D = torch.diag(self.eigval_.exp())
        U, _ = torch.linalg.qr(torch.cat((self.eigvec_.unsqueeze(1),
                                          torch.rand(self.eigvec_.shape[0], self.eigvec_.shape[0]-1).to(x.device)), dim=1))
        return nn.functional.linear(x, torch.mm(U.transpose(1, 0), torch.mm(D, U)).to(x.device))

    # Params
    @property
    def eigval(self):
        return self.eigval_

    @eigval.setter
    def eigval(self, value):
        self.eigval_ = nn.Parameter(value, requires_grad=False)

    # Params
    @property
    def eigvec(self):
        return self.eigvec_

    @eigvec.setter
    def eigvec(self, value):
        self.eigvec_ = nn.Parameter(value, requires_grad=False)


class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)


class MatrixExponential(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)


class SymmetricPositive(nn.Module):
    def __init__(self, in_features):

        super(SPD, self).__init__()

        self.spd_ = nn.Linear(in_features, in_features, bias=False)
        P.register_parametrization(self.spd_, "weight", Symmetric())
        P.register_parametrization(self.spd_, "weight", MatrixExponential())

    def forward(self, x):
        return self.spd_(x)
