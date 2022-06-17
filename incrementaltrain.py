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
from src.parametrization import SPD, Diagonal, Spherical, Fixed
from src.dynamics_second import DynamicsSecond
from src.dynamics_first import DynamicsFirst
from src.trainer import Trainer


# User input
dataset = sys.argv[1] if len(sys.argv) > 1 else "Angle"
load = sys.argv[2].lower() in ['true', '1', 't', 'y', 'yes',
                               'load'] if len(sys.argv) > 2 else False

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
# approximator = KernelMachine(dim, 1000, 1, length=0.3)
approximator = FeedForward(dim, [128, 128, 128], 1)
# layers = nn.ModuleList()
# layers.append(KernelMachine(dim, 250, dim+1, length=0.45))
# for i in range(2):
#     layers.append(CouplingLayer(dim+1, 250, i % 2, 0.45))
# approximator = nn.Sequential(*(layers[i] for i in range(len(layers))))

# Embedding
embedding = Embedding(approximator)
# embedding.apply(embedding.init_weights)

# Attractor
attractor = X[-1, :dim]

# Stiffness
stiffness = SPD(dim)
# stiffness.spherical = (torch.tensor(1.), False)

# Dynamics first order
ds1 = DynamicsFirst(attractor, stiffness, embedding).to(device)

# Trainer first order
trainer = Trainer(ds1, X[:, :dim], X[:, dim:])
trainer.optimizer = torch.optim.Adam(
    trainer.model.parameters(), lr=1e-3, weight_decay=1e-8)
trainer.loss = torch.nn.MSELoss()
# trainer.loss = torch.nn.SmoothL1Loss()
trainer.options(normalize=False, shuffle=True, print_loss=True,
                epochs=10000, load_model=(dataset if load else None))
trainer.train()
trainer.save(dataset+"1")

# Dynamics second order
K = stiffness.spd.weight.data
L, V = torch.linalg.eig(K)
# stiffness = Fixed(dim)
# stiffness.fixed = K
dissipation = Fixed(dim)
dissipation.fixed = torch.mm(torch.real(V), torch.mm(torch.diag(
    2*torch.sqrt(torch.real(L))), torch.real(V.transpose(1, 0))))
ds2 = DynamicsSecond(attractor, stiffness,
                     dissipation, embedding).to(device)

# Trainer second order
trainer.model = ds2
trainer.input = X
trainer.target = Y
trainer.optimizer = torch.optim.Adam(
    trainer.model.parameters(), lr=1e-4, weight_decay=1e-8)
trainer.options(epochs=1000, load_model=None)
# trainer.train()

# Save model
trainer.save(dataset+"2")
