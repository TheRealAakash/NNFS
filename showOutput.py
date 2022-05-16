import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

from Activations import *
from Optimizers import *
from network import Network
import sklearn
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'

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
while True:
    network.train(X, y, n_epochs=1, print_every=100)
    out = network.predict(X_test) ** 0.3
    out = np.clip(out, 0, 0.9)
    plt.clf()
    plt.scatter(X_test[:, 0], X_test[:, 1], s=9, c=out)
    ax = plt.gca()
    plt.axis('off')
    ax.margins(x=0)
    plt.draw()
    plt.pause(0.00001)


# plt.scatter(X_test[:, 0], X_test[:, 1], c=out)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()
