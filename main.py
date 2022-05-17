from Network import *
from nnfs.datasets import sine_data, spiral_data
import nnfs

nnfs.init()

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)
model = Model()
model.add(LayerDense(2, 512,
                     weight_regularizer_l2=5e-4,
                     bias_regularizer_l2=5e-4))
model.add(ActivationReLU())
model.add(LayerDropout(0.1))
model.add(LayerDense(512, 3))
model.add(ActivationSoftmax())

model.set(
    loss=LossCategoricalCrossentropy(),
    optimizer=OptimizerAdam(learning_rate=0.05, decay=5e-7),
    accuracy=AccuracyCategorical(),
)
model.finalize()
model.train(X, y, epochs=10000, print_every=100, validationData=(X_test, y_test))
