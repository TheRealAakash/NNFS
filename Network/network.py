import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from .Activations import *
from .Layers import *
from .Loss import *
from .Optimizers import *
import matplotlib.pyplot as plt


class Network:
    def __init__(self, optimizer):
        self.layers = []
        self.optimizer = optimizer
        self.lastLayer = None
        self.epochs = 0

        self.losses = []
        self.accuracies = []

    def addDense(self, n_outputs, n_inputs=None, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        if self.lastLayer is None:
            self.lastLayer = LayerDense(n_inputs, n_outputs, weight_regularizer_l1, weight_regularizer_l2, bias_regularizer_l1, bias_regularizer_l2)
        else:
            self.lastLayer = LayerDense(self.lastLayer.n_neurons, n_outputs, weight_regularizer_l1, weight_regularizer_l2, bias_regularizer_l1, bias_regularizer_l2)
        self.layers.append(self.lastLayer)

    def addActivation(self, activationFunction):
        self.layers.append(activationFunction())

    def addDropout(self, dropout):
        self.layers.append(LayerDropout(dropout))

    def setLoss(self, lossFunction):
        self.loss = lossFunction

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def trainEpoch(self, X, y):
        for layer in self.layers:
            out = layer.forward(X)
            X = layer.output
        loss = self.loss.forward(X, y)
        for layer in self.layers:
            if isinstance(layer, LayerDense):
                loss += self.loss.loss.regularization_loss(layer)

        self.loss.backward(self.loss.output, y)
        dinputs = self.loss.dinputs
        for layer in reversed(self.layers):
            layer.backward(dinputs)
            dinputs = layer.dinputs

        self.optimizer.pre_update_params()
        for layer in self.layers:
            if type(layer) == LayerDense:
                self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

        predictions = np.argmax(self.loss.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)
        return loss, accuracy

    def train(self, X, y, n_epochs, print_every=100, graphEvery=10, printOut=True):
        for epoch in range(n_epochs):
            self.epochs += 1

            loss, accuracy = self.trainEpoch(X, y)
            self.losses.append(loss)
            self.accuracies.append(accuracy)
            if not self.epochs % print_every and printOut:
                print(f"Epoch: {self.epochs}\n"
                      + f"Accuracy: {accuracy:.3f}\n"
                      + f"Loss: {loss:.3f}\n"
                      + f"Learning Rate: {self.optimizer.current_learning_rate}\n")
            if self.epochs % graphEvery == 0:
                plt.clf()
                # plt.scatter(range(1, self.epochs + 1), self.losses, color='red')
                plt.plot(range(1, self.epochs + 1), self.losses, color='red')
                plt.plot(range(1, self.epochs + 1), self.accuracies, color='green')
                plt.draw()
                plt.pause(0.00001)

    def predict(self, X):
        for layer in self.layers:
            layer.forward(X)
            X = layer.output
        self.loss.activation.forward(X)
        return self.loss.activation.output
