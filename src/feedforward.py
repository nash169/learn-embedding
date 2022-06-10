#!/usr/bin/env python

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, in_features, structure, out_features, encoding=None):
        super(FeedForward, self).__init__()

        self.dim_ = in_features * encoding * 2 if encoding is not None else in_features

        structure = [self.dim_] + structure

        layers = nn.ModuleList()

        for i, _ in enumerate(structure[:-1]):
            layers.append(nn.Linear(structure[i], structure[i+1]))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(structure[-1], out_features))

        self.net_ = nn.Sequential(*(layers[i] for i in range(len(layers))))

    def forward(self, x):
        return self.net_(x if self.dim_ == x.shape[1] else self.encoding(x))

    def encoding(self, x):
        L = 0
        y = torch.zeros((x.shape[0], self.dim_)).to(x.device)
        for i in range(0, self.dim_, x.shape[1]*2):
            y[:, i: i + x.shape[1]
              ] = torch.sin(torch.pow(torch.tensor(2), L) * torch.pi * x)
            y[:, i + x.shape[1]: i + 2*x.shape[1]
              ] = torch.cos(torch.pow(torch.tensor(2), L) * torch.pi * x)
            L += 1
        return y
