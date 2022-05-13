import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Activations import *
from Layers import *
from Loss import *

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()
dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()
loss_function = LossCategoricalCrossentropy()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])
loss = loss_function.calculate(activation2.output, y)
print(loss)

predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.arxmax(y, axis=1)

accuracy = np.mean(predictions == y)

print(f"Accuracy: {accuracy}")
