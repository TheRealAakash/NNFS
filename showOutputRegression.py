import time

import cv2
import nnfs
from nnfs.datasets import spiral_data, sine_data

from Network.Activations import *
from Network.Optimizers import *
from Network.Layers import *
from Network.Accuracies import *
from Network.BaseModel import Model
import sklearn
import matplotlib.pyplot as plt

nnfs.init()
X, y = sine_data()
X, y = sklearn.utils.shuffle(X, y)
X_test, y_test = sine_data()
# network.train(X, y, n_epochs=100, print_every=100)

model = Model()
model.add(LayerDense(1, 128, weight_regularizer_l1=5e-4, bias_regularizer_l1=5e-4))
model.add(ActivationReLU())
model.add(LayerDropout(0.1))
model.add(LayerDense(128, 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 1))
model.add(ActivationLinear())
model.set(loss=LossMeanSquaredError(), optimizer=OptimizerAdam(learning_rate=0.005, decay=5e-5), accuracy=AccuracyCategorical())
model.finalize()

SIZE = 300
minX = -1
maxX = 1
minY = -1
maxY = 1
SCALE_FACTOR = 3
vidout = cv2.VideoWriter(f'Videos/output{int(time.time())}.mp4', cv2.VideoWriter_fourcc(*"H264"), 60, (SIZE * SCALE_FACTOR, SIZE * SCALE_FACTOR))

batchSize = 512
print(X.shape)
X = list(X)
batchX = []
for i in range(0, len(X), batchSize):
    batchX.append(np.array(X[i:i + batchSize]))

y = list(y)
batchY = []
for i in range(0, len(y), batchSize):
    batchY.append(np.array(y[i:i + batchSize]))
X = np.array(X)
y = np.array(y)
while True:
    for x_batch, y_batch in zip(batchX, batchY):
        plt.clf()
        model.train(x_batch, y_batch, epochs=1, print_every=100)
        y_pred = model.predict(X_test)
        plt.plot(X_test, y_pred, c='b')
        plt.plot(X_test, y_test, c='g')
        # plt.scatter(x_batch, y_batch, s=4, c='r')
        plt.draw()
        plt.pause(0.00001)
vidout.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("Done")
# plt.scatter(X_test[:, 0], X_test[:, 1], c=out)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()
