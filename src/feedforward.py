#!/usr/bin/env python

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, in_features, structure, out_features):
        super(FeedForward, self).__init__()

        structure = [in_features] + structure

        layers = nn.ModuleList()

        for i, _ in enumerate(structure[:-1]):
            layers.append(nn.Linear(structure[i], structure[i+1]))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(structure[-1], out_features))

        self.net_ = nn.Sequential(*(layers[i] for i in range(len(layers))))

    def forward(self, x):
        return self.net_(x)
