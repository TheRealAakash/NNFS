import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Network.Activations import *
from Network.Layers import *
from Network.Loss import *
from Network.Optimizers import *

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = ActivationReLU()
dropout1 = LayerDropout(0.1)
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
    dropout1.forward(activation1.output)
    dense2.forward(activation1.output)
    data_loss = loss_activation.forward(dense2.output, y)

    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)

    loss = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    if not epoch % 100:
        print(f"Epoch: {epoch}\n"
              + f"Accuracy: {accuracy:.3f}\n"
              + f"Loss: {loss:.3f}\n"
              + f"Data Loss: {data_loss:.3f}\n"
              + f"Regularization Loss: {regularization_loss:.3f}\n"
                f"Learning Rate: {optimizer.current_learning_rate}\n")

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
