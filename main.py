import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Activations import *
from Layers import *
from Loss import *
from Optimizers import *

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 64)
activation1 = ActivationReLU()
dense2 = LayerDense(64, 3)

loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()

# optimizer = OptimizerSGD(learning_rate=1, decay=1e-3, momentum=0.9)
# optimizer = OptimizerAdaGrad(learning_rate=1, decay=1e-4)
# optimizer = OptimizerRMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = OptimizerAdam(learning_rate=0.05, decay=5e-7)

N_EPOCHS = 10001

for epoch in range(N_EPOCHS):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # print("Loss:", loss)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    if not epoch % 100:
        print(f"Epoch: {epoch}\n"
              + f"Accuracy: {accuracy:.3f}\n"
              + f"Loss: {loss:.3f}\n"
                f"Learning Rate: {optimizer.current_learning_rate}\n")

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
