from manim import *
import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

epsilon = 1e-7


def fPrime(f, x):
    return (f(x + epsilon) - f(x)) / epsilon


def fRotateAround(x):
    return x * x


def fRotate(x):
    return math.sin(x)


def newtonsMethodIntersection(f1, f2, guess, numIterations=10):
    if numIterations > 0:
        return newtonsMethodIntersection(f1, f2, guess - (f1(guess) - f2(guess)) / fPrime(lambda x: f1(x) - f2(x), guess), numIterations - 1)
    else:
        return guess


def line(x, slope, pointx, pointy):
    return slope * (x - pointx) + pointy


# calculate the volume of revolution of fRotate, rotating around fRotateAround
minX = 0.0001
maxX = 0.9
linspace = np.linspace(minX, maxX, 30)
deltaX = linspace[1] - linspace[0]
approx = 0
fig = plt.figure()
ax = plt.axes(projection='3d')
plt.plot(linspace, fRotateAround(linspace), color='red')
plt.plot(linspace, list(map(fRotate, list(linspace))), color='green')
for pt in linspace:
    slope = fPrime(fRotateAround, pt)
    normalSlope = -1 / slope
    intersection = newtonsMethodIntersection(fRotate, lambda x: line(x, normalSlope, pt, fRotateAround(pt)), pt)
    #  plt.scatter(intersection, fRotate(intersection), color='blue')
    # plt.scatter(x=pt, y=fRotateAround(pt), s=1, color='blue')
    if pt < intersection:
        plt.plot((pt, intersection), (fRotateAround(pt), line(intersection, normalSlope, pt, fRotateAround(pt))), color='blue')
    else:
        plt.plot((intersection, pt), (line(intersection, normalSlope, pt, fRotateAround(pt)), fRotateAround(pt)), color='blue')

    intersectionLength = math.sqrt((intersection - pt) ** 2 + (fRotate(intersection) - fRotateAround(pt)) ** 2)
    # get circle around (pt, fRotateAround(pt))

    space = np.linspace(0, 2 * math.pi, 4)
    center = (pt, fRotateAround(pt))
    radius = intersectionLength
    for point in space:
        x = center[0] + radius * math.cos(point)
        z = radius * math.sin(point)
        ax.scatter3D(x, center[1], z, color='blue')

approx += ((intersectionLength) ** 2) * math.pi * deltaX

print(approx)
plt.show()
