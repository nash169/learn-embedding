#!/usr/bin/env python

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import copy

from src.kernel_machine import KernelMachine
from src.coupling_layer import CouplingLayer
from src.feedforward import FeedForward
from src.embedding import Embedding
from src.parametrization import SPD, Diagonal, Spherical, Fixed, Rotation
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
data = np.loadtxt(os.path.join('trainset', '{}.csv'.format(dataset)))
dim = int(data.shape[1]/3)

# State (pos,vel)
X = torch.from_numpy(data[:, :2*dim]).float().to(device)

# Output (acc)
Y = torch.from_numpy(data[:, 2*dim:]).float().to(device)

# Attractor
attractor = X[-1, :dim]

# Dynamics First
approximator_1 = FeedForward(dim, [64], 1)
embedding_1 = Embedding(approximator_1)
stiffness_1 = SPD(dim)
stiffness_1.vec_ = nn.Parameter(torch.tensor([1.0, 0.0]))
lambda_1 = -4
stiffness_1.eig_ = nn.Parameter(lambda_1 * torch.ones(2))

ds1 = DynamicsFirst(attractor, stiffness_1, embedding_1).to(device)
ds1.load_state_dict(torch.load(os.path.join(
    'models', '{}.pt'.format(dataset+"1")), map_location=torch.device(device)))
ds1.eval()

# Dynamics second
approximator_2 = FeedForward(dim, [64], 1)  # copy.deepcopy(approximator_1)
embedding_2 = Embedding(approximator_2)
stiffness_2 = SPD(dim)  # copy.deepcopy(stiffness_1)
dissipation = SPD(dim)  # copy.deepcopy(stiffness_1)

ds = DynamicsSecond(attractor, stiffness_2,
                    dissipation, embedding_2).to(device)
# ds.velocity_ = ds1

# Adjust first DS parameters
# stiffness_1.eig_ = nn.Parameter(lambda_1 * torch.ones(2).to(device))
# stiffness_1.vec_ = nn.Parameter(torch.tensor([1.0, 0.0]).to(device))
# embedding_1.apply(embedding_1.init_weights)
for param in ds1.parameters():
    param.requires_grad = False

# # Fix stiffness hyperparameters
# for param in stiffness.parameters():
#     param.requires_grad = False

# # Fix embedding hyperparameters
# for param in embedding.parameters():
#     param.requires_grad = False

# Trainer
trainer = Trainer(ds, X, Y)

# Set trainer optimizer (this is not very clean)
trainer.optimizer = torch.optim.Adam(
    trainer.model.parameters(), lr=1e-2, weight_decay=1e-8)

# Set trainer loss
# trainer.loss = torch.nn.MSELoss()
trainer.loss = torch.nn.SmoothL1Loss()

# Set trainer options
trainer.options(normalize=False, shuffle=True, print_loss=True,
                epochs=10000, load_model=(dataset+"2" if load else None))

# Train model
trainer.train()

# Save model
trainer.save(dataset+"2")
