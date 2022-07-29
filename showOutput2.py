import time

import cv2
import nnfs
from nnfs.datasets import spiral_data

from Network.Activations import *
from Network.Optimizers import *
from Network.Layers import *
from Network.Accuracies import *
from Network.BaseModel import Model
import sklearn

nnfs.init()
X, y = spiral_data(samples=1000, classes=3)
X, y = sklearn.utils.shuffle(X, y)
ident = np.eye(3)
ident[2][0] = 1
ident[2][1] = 1
ident[2][2] = 0
y_col = ident[y]
# network.train(X, y, n_epochs=100, print_every=100)

model = Model()
model.add(LayerDense(2, 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ActivationReLU())
model.add(LayerDense(128, 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ActivationReLU())
model.add(LayerDense(128, 3, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ActivationSoftmax())
model.set(loss=LossCategoricalCrossentropy(), optimizer=OptimizerAdam(learning_rate=0.005, decay=5e-6),
          accuracy=AccuracyCategorical())
model.finalize()

SIZE = 250
SIZE_TARGET = 500
minX = -1
maxX = 1
minY = -1
maxY = 1
X_test = np.linspace(-1, 1, SIZE)
X_test = np.array([[c1, c2] for c1 in X_test for c2 in X_test])
linspace = np.linspace(-1, 1, SIZE)
batchSize = 16
frameNum = 0
image = np.zeros((SIZE, SIZE, 3))
while True:
    for i in range(5):
        x_batch, y_batch = spiral_data(samples=batchSize, classes=3)
        x_batch, y_batch = sklearn.utils.shuffle(x_batch, y_batch)
        model.train(x_batch, y_batch, epochs=1, print_every=100)
    out = model.predict(X_test) ** 0.3
    for i in range(len(X_test)):
        x = X_test[i, 0]
        y_c = X_test[i, 1]
        x = (x - minX) / (maxX - minX) * SIZE
        y_c = (y_c - minY) / (maxY - minY) * SIZE
        x = int(x) - 1
        y_c = int(y_c) - 1
        col = out[i]
        r = col[2]
        image[y_c, x] = max(out[i][0], r), max(out[i][1], + r), 0

    for point, col in zip(X, y_col):
        x = (point[0] - minX) / (maxX - minX) * SIZE
        y_c = (point[1] - minY) / (maxY - minY) * SIZE
        x = int(x) - 1
        y_c = int(y_c) - 1
        image[y_c, x] = col
    # for point, col in zip(x_batch, y_batch):
    #     x = (point[0] - minX) / (maxX - minX) * SIZE
    #     y_c = (point[1] - minY) / (maxY - minY) * SIZE
    #     x = int(x) - 1
    #     y_c = int(y_c) - 1
    #     image[y_c, x] = (1, 1, 1)
    # cv2.circle(image, (x, y_c), 1, (int(col[0]), int(col[1]), int(col[2])), -1)
    # crop image to remove black borders
    output = image[2:-2, 2:-2]
    output = cv2.resize(output, (SIZE_TARGET, SIZE_TARGET), interpolation=cv2.INTER_NEAREST)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output = np.array(output * 255, np.uint8)

    cv2.imshow("Model", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    print(frameNum)
    frameNum += 1
