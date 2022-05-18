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
model.add(LayerDense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ActivationReLU())
model.add(LayerDropout(0.1))
model.add(LayerDense(64, 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ActivationReLU())
model.add(LayerDropout(0.1))
model.add(LayerDense(128, 3))
model.add(ActivationSoftmax())
model.set(loss=LossCategoricalCrossentropy(), optimizer=OptimizerAdam(learning_rate=0.05, decay=5e-7), accuracy=AccuracyCategorical())
model.finalize()

SIZE = 300
minX = -1
maxX = 1
minY = -1
maxY = 1
SCALE_FACTOR = 3
X_test = np.linspace(-1, 1, SIZE)
X_test = np.array([[c1, c2] for c1 in X_test for c2 in X_test])
vidout = cv2.VideoWriter(f'Videos/output{int(time.time())}.mp4', cv2.VideoWriter_fourcc(*"H265"), 30, (SIZE * SCALE_FACTOR, SIZE * SCALE_FACTOR))

batchSize = 3000
print(X.shape)
X = list(X)
batchX = []
for i in range(0, len(X), batchSize):
    batchX.append(np.array(X[i:i + batchSize]))

y = list(y)
batchY = []
for i in range(0, len(y), batchSize):
    batchY.append(np.array(y[i:i + batchSize]))
while True:
    for x_batch, y_batch in zip(batchX, batchY):
        image = np.zeros((SIZE, SIZE, 3))
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
        image = image[2:-2, 2:-2]
        image = cv2.resize(image, (SIZE * SCALE_FACTOR, SIZE * SCALE_FACTOR), interpolation=cv2.INTER_NEAREST)
        image2 = np.array(image * 255, np.uint8)
        vidout.write(image2)
        cv2.imshow("image", image2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vidout.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("Done")
# plt.scatter(X_test[:, 0], X_test[:, 1], c=out)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()
