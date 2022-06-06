#!/usr/bin/env python

import os
import sys
import numpy as np
import torch

from src.kernel import Kernel
from src.feedforward import FeedForward
from src.embedding import Embedding
from src.parametrization import SPD, Diagonal
from src.dynamics_second import DynamicsSecond
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
kernel = Kernel(dim, 1000, 1, length=0.45)
# feedforward = FeedForward(dim, [100, 100], 1)

# Embedding
embedding = Embedding(kernel)


def metric(y):
    g = torch.eye(y.shape[1])
    return g.repeat(y.shape[0], 1, 1).to(device)


embedding.metric = metric

# Attractor
attractor = X[-1, :dim]

# Stiffness
stiffness = Diagonal(dim)

# Dissipation
dissipation = Diagonal(dim)

# Dynamics
ds = DynamicsSecond(attractor, stiffness,
                    dissipation, embedding).to(device)

# Create trainer
trainer = Trainer(ds, X, Y)

# Set trainer optimizer (this is not very clean)
trainer.optimizer = torch.optim.Adam(
    trainer.model.parameters(), lr=1e-2,  weight_decay=1e-8)

# Set trainer loss
trainer.loss = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss()

# Set trainer options
trainer.options(normalize=False, shuffle=True, print_loss=True,
                epochs=1000, load_model=(dataset if load else None))

# Train model
trainer.train()

# Save model
trainer.save(dataset)
