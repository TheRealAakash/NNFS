import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Activations import *
from Layers import *
from Loss import *
from Optimizers import *
from network import Network

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
network = Network(optimizer=OptimizerAdam(learning_rate=0.05, decay=5e-7))
network.addDense(64, 2)
network.addActivation(ActivationReLU)
network.addDense(3)
network.setLoss(ActivationSoftmaxLossCategoricalCrossentropy())
network.train(X, y, n_epochs=10001, print_every=100)
