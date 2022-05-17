import numpy as np
import nnfs
from nnfs.datasets import sine_data
from Network.Activations import *
from Network.Layers import *
from Network.Loss import *
from Network.Optimizers import *
import matplotlib.pyplot as plt

nnfs.init()

X, y = sine_data()

dense1 = LayerDense(1, 64)
activation1 = ActivationReLU()
dense2 = LayerDense(64, 64)
activation2 = ActivationReLU()
dense3 = LayerDense(64, 1)
activation3 = ActivationLinear()
loss_function = LossMeanSquaredError()
# optimizer = OptimizerSGD(learning_rate=1, decay=1e-3, momentum=0.9)
# optimizer = OptimizerAdaGrad(learning_rate=1, decay=1e-4)
# optimizer = OptimizerRMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = OptimizerAdam(learning_rate=0.005, decay=1e-3)

accuracyPrecision = np.std(y) / 250

N_EPOCHS = 10001

for epoch in range(N_EPOCHS):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    data_loss = loss_function.calculate(activation3.output, y)

    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2) + loss_function.regularization_loss(dense3)

    loss = data_loss + regularization_loss

    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracyPrecision)
    if not epoch % 100:
        print(f"Epoch: {epoch}\n"
              + f"Accuracy: {accuracy:.3f}\n"
              + f"Loss: {loss:.3f}\n"
              + f"Data Loss: {data_loss:.3f}\n"
              + f"Regularization Loss: {regularization_loss:.3f}\n"
                f"Learning Rate: {optimizer.current_learning_rate}\n")

    # Backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

plt.plot(X, activation3.output, color='red')
plt.show()
