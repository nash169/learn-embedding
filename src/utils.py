#!/usr/bin/env python

import torch


def linear_map(x, xmin, xmax, ymin, ymax):
    m = (ymin - ymax)/(xmin-xmax)
    q = ymin - m*xmin

    y = m*x + q

    return y


def squared_exp(x, y, sigma=1, eta=1):
    return eta*torch.exp(-0.5*torch.sum((x-y).pow(2), dim=1) / sigma ** 2).unsqueeze(1)


def infty_exp(x, y, r=1, a=1, b=2):
    return torch.exp(a/(b*torch.pow(torch.norm(x-y, dim=1) - r, b))).unsqueeze(1)
