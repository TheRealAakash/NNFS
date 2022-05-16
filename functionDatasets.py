import math

import numpy as np


def mandelbrot(numSamples):
    X = []
    y = []

    for i in range(numSamples):
        numReal = (np.random.random() * 4) - 2
        numImag = (np.random.random() * 4) - 2
        X.append([numReal / 2, numImag / 2])
        num = complex(numReal, numImag)
        good = True
        for j in range(50):
            num = num * num + complex(numReal, numImag)
            if abs(num) > 2:
                y.append([1, 0])
                good = False
                break
        if abs(num) < 2 and good:
            y.append([0, 1])

    return np.array(X), np.array(y)


def test():
    import matplotlib.pyplot as plt
    X, y = mandelbrot(100000)
    plt.scatter(X[:, 0], X[:, 1], s=1, c=np.argmax(y, axis=1))
    plt.show()


if __name__ == '__main__':
    test()
