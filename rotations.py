from manim import *
import math


def f1(x):
    return x * x


def f2(x):
    return math.sin(x)


start = 0
stop = 2


class Anim(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        axes.plot(f1, (0., 1.), color=RED)

