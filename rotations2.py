from manim import *
import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import tqdm

epsilon = 1e-7
config.disable_caching = True


def fPrime(f, x):
    return (f(x + epsilon) - f(x)) / epsilon


def fRotateAround(x):
    return math.sqrt(x)


def fRotate(x):
    return math.cos(x)


def newtonsMethodIntersection(f1, f2, guess, numIterations=10):
    if numIterations > 0:
        return newtonsMethodIntersection(f1, f2, guess - (f1(guess) - f2(guess)) / fPrime(lambda x: f1(x) - f2(x), guess), numIterations - 1)
    else:
        return guess


def line(x, slope, pointx, pointy):
    return slope * (x - pointx) + pointy


class Rotation(ThreeDScene):
    def construct(self):
        function1 = FunctionGraph(fRotate, (0.1, 1), color=RED)
        function2 = FunctionGraph(fRotateAround, (0.1, 1), color=BLUE)
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        axes = ThreeDAxes()
        self.play(Create(axes))
        self.play(Create(function1), Create(function2))
        anims0 = []
        anims = []
        space = np.linspace(0.1, 1, 100)
        for pt in space:
            pos1 = (pt, fRotateAround(pt), 0)
            slope = fPrime(fRotateAround, pt)
            normalSlope = -1 / slope
            intersection = newtonsMethodIntersection(lambda x: fRotate(x), lambda x: line(x, normalSlope, pos1[0], pos1[1]), pos1[0])
            pos2 = (intersection, fRotate(intersection), 0)

            pos1Tan = (0.3, line(0.3, slope, pos1[0], pos1[1]), 0)
            pos2Tan = (0.8, line(0.8, slope, pos1[0], pos1[1]), 0)

            tanLine = Line(pos1Tan, pos2Tan, color=GREEN)
            # self.play(Create(tanLine))

            intersectionLine = Line(pos1, pos2, color=YELLOW)

            anims0.append(Create(intersectionLine))
            anims.append(Rotate(intersectionLine, angle=2 * PI, axis=tanLine.get_vector(), about_point=tanLine.get_start(), rate_func=linear))
            count = 50
            self.add(intersectionLine)
            for i in tqdm.tqdm(range(count)):
                self.add(intersectionLine.copy().rotate(360 * DEGREES * i / count, axis=tanLine.get_vector(), about_point=tanLine.get_start()))
            # anims.append(Rotate(intersectionLine, angle=2 * PI, axis=tanLine.get_vector(), about_point=tanLine.get_start(), rate_func=linear))
        # self.play(*anims0)
        # for i in range(10):
        self.begin_ambient_camera_rotation(rate=0.2)
       #  self.play(self.camera.rotate(2*PI, axis=UP, about_point=ORIGIN, rate_func=linear), run_time=5)
        self.wait(10)
