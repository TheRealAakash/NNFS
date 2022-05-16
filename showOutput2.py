import cv2
import nnfs
from functionDatasets import mandelbrot
from Activations import *
from Optimizers import *
from network import Network

nnfs.init()
X, y = mandelbrot(numSamples=1000)
network = Network(optimizer=OptimizerAdam(learning_rate=0.05, decay=5e-6))
network.addDense(64, 2)
network.addActivation(ActivationReLU)
network.addDense(256)
network.addActivation(ActivationReLU)
network.addDense(256)
network.addActivation(ActivationReLU)
network.addDense(256)
network.addActivation(ActivationReLU)
network.addDense(256)
network.addActivation(ActivationReLU)
network.addDense(256)
network.addActivation(ActivationReLU)
network.addDense(256)
network.addActivation(ActivationReLU)
network.addDense(256)
network.addActivation(ActivationReLU)
network.addDense(256)
network.addActivation(ActivationReLU)
network.addDense(2)
network.setLoss(ActivationSoftmaxLossCategoricalCrossentropy())
# network.train(X, y, n_epochs=100, print_every=100)

SIZE = 300
minX = -1
maxX = 1
minY = -1
maxY = 1
X_test = np.linspace(-1, 1, SIZE)
X_test = np.array([[c1, c2] for c1 in X_test for c2 in X_test])
vidout = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (SIZE * 3, SIZE * 3))

while True:
    X, y = mandelbrot(numSamples=8000)
    network.train(X, y, n_epochs=1, graphEvery=25, print_every=100)
    if network.epochs % 50:
        continue
    image = np.zeros((SIZE, SIZE, 3))
    out = network.predict(X_test)
    for i in range(len(X_test)):
        x = X_test[i, 0]
        y_c = X_test[i, 1]
        x = (x - minX) / (maxX - minX) * SIZE
        y_c = (y_c - minY) / (maxY - minY) * SIZE
        x = int(x) - 1
        y_c = int(y_c) - 1
        image[y_c, x] = (out[i][1], out[i][1], 0)
    # for point, col in zip(X, y_col):
    #     x = (point[0] - minX) / (maxX - minX) * SIZE
    #     y_c = (point[1] - minY) / (maxY - minY) * SIZE
    #     x = int(x) - 1
    #     y_c = int(y_c) - 1
    #     # image[y_c, x] = col[1], 0, 0
    # for point, col in zip(x_batch, y_batch):
    #     x = (point[0] - minX) / (maxX - minX) * SIZE
    #     y_c = (point[1] - minY) / (maxY - minY) * SIZE
    #     x = int(x) - 1
    #     y_c = int(y_c) - 1
    #     image[y_c, x] = (1, 1, 1)
    # cv2.circle(image, (x, y_c), 1, (int(col[0]), int(col[1]), int(col[2])), -1)
    # crop image to remove black borders
    image = image[2:-2, 2:-2]
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
