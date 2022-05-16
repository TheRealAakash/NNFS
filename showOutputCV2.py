import cv2
import nnfs
from nnfs.datasets import spiral_data

from Activations import *
from Optimizers import *
from network import Network
import sklearn

nnfs.init()
X, y = spiral_data(samples=200, classes=3)
X, y = sklearn.utils.shuffle(X, y)

y_col = np.eye(3)[y]
network = Network(optimizer=OptimizerAdam(learning_rate=0.05, decay=5e-7))
network.addDense(64, 2)
network.addActivation(ActivationReLU)
network.addDense(3)
network.setLoss(ActivationSoftmaxLossCategoricalCrossentropy())
network.train(X, y, n_epochs=100, print_every=100)

SIZE = 100
minX = -1
maxX = 1
minY = -1
maxY = 1
X_test = np.linspace(-1, 1, SIZE)
X_test = np.array([[c1, c2] for c1 in X_test for c2 in X_test])
vidout = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (SIZE * 4, SIZE * 4))
while True:
    image = np.zeros((SIZE, SIZE, 3))
    network.train(X, y, n_epochs=1, print_every=100)
    out = network.predict(X_test) ** 0.3
    out = np.clip(out, 0, 0.9)
    for i in range(len(X_test)):
        x = X_test[i, 0]
        y_c = X_test[i, 1]
        x = (x - minX) / (maxX - minX) * SIZE
        y_c = (y_c - minY) / (maxY - minY) * SIZE
        x = int(x) - 1
        y_c = int(y_c) - 1
        image[y_c, x] = out[i]
    for point, col in zip(X, y_col):
        x = (point[0] - minX) / (maxX - minX) * SIZE
        y_c = (point[1] - minY) / (maxY - minY) * SIZE
        x = int(x) - 1
        y_c = int(y_c) - 1
        image[y_c, x] = col
        # cv2.circle(image, (x, y_c), 1, (int(col[0]), int(col[1]), int(col[2])), -1)
    # crop image to remove black borders
    image = image[2:-2, 2:-2]
    image = cv2.resize(image, (SIZE * 4, SIZE * 4), interpolation=cv2.INTER_NEAREST)
    image2 = np.array(image * 255, np.uint8)
    vidout.write(image2)
    cv2.imshow("image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vidout.release()
cv2.destroyAllWindows()
# plt.scatter(X_test[:, 0], X_test[:, 1], c=out)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()
