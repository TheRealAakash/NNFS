import numpy as np
from manim import *
import math


def f1(x):
    return x * x


def f2(x):
    return math.sin(x)


start = 0
stop = 0.5


class Anim(ThreeDScene):
    def construct(self):
        function = FunctionGraph(f1, (start, stop), color=RED)
        function2 = FunctionGraph(f2, (start, stop), color=RED)

        self.play(Create(function))
        self.play(Create(function2))

        points = np.linspace(start, stop, 10)
        for point in points:
            LineNormalToFunction1AtPoint = Line(
                function.get_point_from_function(point),
                function2.get_point_from_function(point),
                color=RED
            )
            self.play(Create(LineNormalToFunction1AtPoint))
            TangentVectorToFunction1AtPoint = Vector(
                function.get_tangent_vector_at_function(point),
                color=RED
            )


            self.play(Rotate(LineNormalToFunction1AtPoint, angle=PI / 2, axis=TangentVectorToFunction1AtPoint))

        self.wait(2)
