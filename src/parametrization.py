#!/usr/bin/env python

import torch
import torch.nn as nn


class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)


class MatrixExponential(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)
