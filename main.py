#!/usr/bin/env python

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dynamics import Dynamics
from src.trainer import Trainer

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data
data = np.loadtxt(os.path.join('data', '{}.csv'.format("Angle")))

# State (pos,vel)
X = torch.from_numpy(data[:, :4]).float().to(device).requires_grad_(True)

# Output (acc)
Y = torch.from_numpy(data[:, 4:6]).float().to(device).requires_grad_(True)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data[:, 0], data[:, 1])

# Create Model
dim = 2
structure = [10, 10]
K = torch.eye(2, 2).to(device)
D = torch.eye(2, 2).to(device)

ds = Dynamics(2, structure).to(device)
ds.dissipation = (D, False)
ds.stiffness = (K, False)

# Create trainer
trainer = Trainer(ds, X, Y)

# Set trainer optimizer (this is not very clean)
trainer.optimizer = torch.optim.Adam(
    trainer.model.parameters(), lr=1e-4,  weight_decay=1e-8)

# Set trainer loss
trainer.loss = torch.nn.SmoothL1Loss()

# Set trainer options
trainer.options(normalize=False, shuffle=True, print_loss=True,
                epochs=100, load_model=None)

# Train model
trainer.train()

# Save model
trainer.save("Angle")
