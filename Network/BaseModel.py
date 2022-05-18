from .Activations import *
from .Optimizers import *
from .Losses import *
from .Layers import *


class Model:
    def __init__(self):
        self.layers = []
        self.trainable_layers = []
        self.finalized = False
        self.softmax_classifier_output = None
        self.epochs = 0

    def add(self, layer):
        self.finalized = False
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy, finalize=False):
        self.finalized = False
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
        if finalize:
            self.finalize()

    def finalize(self):
        self.trainable_layers = []
        self.input_layer = LayerInput()

        layer_count = len(self.layers)
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])
        self.loss.remember_trainable_layers(self.trainable_layers)
        self.finalized = True

        if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(self.loss, LossCategoricalCrossentropy):
            self.softmax_classifier_output = ActivationSoftmaxLossCategoricalCrossentropy()

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        layer = self.input_layer
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def train(self, X, y, *, epochs=1, print_every=100, finalize=False, validationData=None):
        if finalize:
            self.finalize()
        if not self.finalized:
            raise Exception("Model must be finalized before training")  # Can auto finalize, but may cause issues
        self.accuracy.init(y)
        for epoch in range(1, epochs + 1):
            self.epochs += 1
            output = self.forward(X, training=True)
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)

            loss = data_loss + regularization_loss

            predictions = self.output_layer_activation.predictions(output)

            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)

            self.optimizer.post_update_params()

            if not self.epochs % print_every:
                print(
                    f"Epoch: {self.epochs}\n" +
                    f"Accuracy: {accuracy:.3f}\n" +
                    f"Loss: {loss:.3f}\n" +
                    f"\tData Loss: {data_loss:.3f}\n" +
                    f"\tRegularization Loss: {regularization_loss:.3f}\n" +
                    f"Learning Rate: {self.optimizer.current_learning_rate}\n"
                )

        if validationData is not None:
            X_val, y_val = validationData
            output = self.forward(X_val, training=False)

            val_loss = self.loss.calculate(output, y_val, include_regularization=False)
            val_predictions = self.output_layer_activation.predictions(output)

            val_accuracy = self.accuracy.calculate(val_predictions, y_val)

            print(
                f"Validation Accuracy: {val_accuracy:.3f}\n" +
                f"Validation Loss: {val_loss:.3f}\n"
            )

    def predict(self, X):
        output = self.forward(X, training=False)
        return output

    def predictProcessed(self, X):
        output = self.forward(X, training=False)
        return self.output_layer_activation.predictions(output)
