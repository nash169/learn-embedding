#!/usr/bin/env python

import os
import sys
import numpy as np
import torch
import torch.nn as nn

from src.kernel_machine import KernelMachine
from src.coupling_layer import CouplingLayer
from src.feedforward import FeedForward
from src.embedding import Embedding
from src.parametrization import SPD, Diagonal, Spherical
from src.dynamics_second import DynamicsSecond
from src.dynamics_first import DynamicsFirst
from src.trainer import Trainer


# User input
dataset = sys.argv[1] if len(sys.argv) > 1 else "Angle"
load = sys.argv[2].lower() in ['true', '1', 't', 'y', 'yes',
                               'load'] if len(sys.argv) > 2 else False
first = False

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data
data = np.loadtxt(os.path.join('data', '{}.csv'.format(dataset)))
dim = int(data.shape[1]/3)

# State (pos,vel)
X = torch.from_numpy(data[:, :2*dim]).float().to(device)

# Output (acc)
Y = torch.from_numpy(data[:, 2*dim:]).float().to(device)

# Function approximator
approximator = KernelMachine(dim, 1000, 1, length=0.1)
# approximator = FeedForward(dim, [10, 10], 1)
# layers = nn.ModuleList()
# layers.append(KernelMachine(dim, 250, dim+1, length=0.45))
# for i in range(2):
#     layers.append(CouplingLayer(dim+1, 250, i % 2, 0.45))
# approximator = nn.Sequential(*(layers[i] for i in range(len(layers))))

# Embedding
embedding = Embedding(approximator)


def metric(y):
    g = torch.eye(y.shape[1])
    return g.repeat(y.shape[0], 1, 1).to(device)


embedding.metric = metric

# Attractor
attractor = X[-1, :dim]

# Stiffness
stiffness = Spherical(dim)

# Dissipation
dissipation = Spherical(dim)

# Dynamics & Trainer
if first:
    ds = DynamicsFirst(attractor, stiffness, embedding).to(device)
    trainer = Trainer(ds, X[:, :dim], X[:, dim:])
else:
    ds = DynamicsSecond(attractor, stiffness,
                        dissipation, embedding).to(device)
    trainer = Trainer(ds, X, Y)

# Set trainer optimizer (this is not very clean)
trainer.optimizer = torch.optim.Adam(
    trainer.model.parameters(), lr=1e-3,  weight_decay=1e-6)

# Set trainer loss
trainer.loss = torch.nn.MSELoss()
# trainer.loss = torch.nn.SmoothL1Loss()

# Set trainer options
trainer.options(normalize=False, shuffle=True, print_loss=True,
                epochs=10000, load_model=(dataset if load else None))

# Train model
trainer.train()

# Save model
trainer.save(dataset)
