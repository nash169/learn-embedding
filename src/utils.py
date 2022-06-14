#!/usr/bin/env python

import os
import torch


# Map dataset in between un upper and lower
# (should be expanded to n-dimension and maybe by default a square box)
def linear_map(x, xmin, xmax, ymin, ymax):
    m = (ymin - ymax)/(xmin-xmax)
    q = ymin - m*xmin

    y = m*x + q

    return y


# RBF kernel
def squared_exp(x, y, sigma=1, eta=1):
    return eta*torch.exp(-0.5*torch.sum((x-y).pow(2), dim=1) / sigma ** 2).unsqueeze(1)


# Control barrier function
def infty_exp(x, y, r=1, a=1, b=2):
    return torch.exp(a/(b*torch.pow(torch.norm(x-y, dim=1) - r, b))).unsqueeze(1)


# Save model/dict
def save(file, model, dict=True):
    assert issubclass(type(model), torch.nn.Module)
    if dict:
        if not os.path.exists('dicts'):
            os.makedirs('dicts')
        torch.save(model.state_dict(), os.path.join(
            'dicts', '{}.pt'.format(file)))
    else:
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(model, os.path.join(
            'models', '{}.pt'.format(file)))


# Load model/dict
def load(file, model, device):
    model.load_state_dict(torch.load(
        os.path.join('models', '{}.pt'.format(file)), map_location=torch.device(device)))
    model.eval()


def load(file, device):
    model = torch.load(
        os.path.join('models', '{}.pt'.format(file)), map_location=torch.device(device))
    model.eval()
    return model
