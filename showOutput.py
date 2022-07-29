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
from manim import *
# manim opengl
from manim.opengl import *

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
model.add(LayerDense(2, 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ActivationReLU())
model.add(LayerDense(128, 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ActivationReLU())
model.add(LayerDense(128, 3, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ActivationSoftmax())
model.set(loss=LossCategoricalCrossentropy(), optimizer=OptimizerAdam(learning_rate=0.005, decay=5e-7),
          accuracy=AccuracyCategorical())
model.finalize()

SIZE = 250
SIZE_TARGET = 500
minX = -1
maxX = 1
minY = -1
maxY = 1
X_test = np.linspace(-1, 1, SIZE)
X_test = np.array([[c1, c2] for c1 in X_test for c2 in X_test])

config.disable_caching = True


class Training(Scene):
    def construct(self):
        image = np.zeros((SIZE_TARGET, SIZE_TARGET, 3), dtype=np.uint8)
        imageMob = ImageMobject(image)
        imageMob.scale(2)
        imageMob.move_to(LEFT * 2)
        self.add(imageMob)
        lossAxis = Axes(
            y_range=[0, 2, 0.5],
            x_range=[0, 100, 100],
            y_axis_config={
                "include_numbers": True,
            },
            axis_config={
                "include_tip": False,
            }
        )
        # rect behind the loss lossGraph

        lossAxis.scale(0.25)
        lossAxis.move_to(RIGHT * 4 + DOWN * 1)
        lossTitle = Text("Loss", font_size=15)
        rect = Rectangle(
            width=lossAxis.x_axis.get_width(),
            height=lossAxis.y_axis.get_height(),
            stroke_width=0,
            fill_color="#2f2f2f",
            fill_opacity=0.5,
        )
        rect.move_to(lossAxis.x_axis.get_left(), LEFT).shift(UP * 0.75)
        lossTitle.move_to(rect.get_center() + UP * 0.5)
        self.add(rect)
        self.add(lossAxis)
        self.add(lossTitle)

        accuracyAxis = Axes(
            y_range=[0, 1, 0.2],
            x_range=[0, 100, 100],
            axis_config={
                "include_tip": False,
            },
            y_axis_config={
                "include_numbers": True,
                "font_size": 50,
            },

        )

        accuracyAxis.scale(0.25)
        accuracyAxis.move_to(RIGHT * 4 + UP * 1)
        rect2 = Rectangle(
            width=accuracyAxis.x_axis.get_width(),
            height=accuracyAxis.y_axis.get_height(),
            stroke_width=0,
            fill_color="#2f2f2f",
            fill_opacity=0.5,
        )
        rect2.move_to(accuracyAxis.x_axis.get_left(), LEFT).shift(UP * 0.75)
        accTitle = Text("Accuracy", font_size=15)

        accTitle.move_to(rect2.get_center() + UP * 0.5)
        self.add(rect2)
        self.add(accuracyAxis)
        self.add(accTitle)

        batchSize = 16
        image = np.zeros((SIZE, SIZE, 3))

        accuracyDot = Dot(color=WHITE, radius=0.05)
        lossDot = Dot(color=WHITE, radius=0.05)
        accuracyDot.move_to(accuracyAxis.coords_to_point(0, 0, 0))
        lossDot.move_to(lossAxis.coords_to_point(0, 0, 0))
        accuracyPath = VMobject(color=GREEN)
        lossPath = VMobject(color=ORANGE)
        self.countAccuracy = 0
        self.countLoss = 0

        def update_path(path):
            previous_path = path.copy()
            previous_path.shift(LEFT * rect2.get_width() / (len(model.history.losses) + 1))
            previous_path.add_points_as_corners([accuracyDot.get_center()])
            previous_path.stretch_to_fit_width(rect2.get_width(), about_edge=RIGHT)
            path.become(previous_path)

        def update_loss_path(path):
            previous_path = path.copy()
            previous_path.shift(LEFT * rect2.get_width() / (len(model.history.losses) + 1))
            previous_path.add_points_as_corners([lossDot.get_center()])
            previous_path.stretch_to_fit_width(rect2.get_width(), about_edge=RIGHT)
            path.become(previous_path)

        def update_dot(dot):
            prev_dot = dot.copy()
            prev_dot.move_to(accuracyAxis.coords_to_point(100, model.history.accuracies[self.countAccuracy]))
            dot.become(prev_dot)
            self.countAccuracy += 1
            if self.countAccuracy == len(model.history.accuracies):
                self.countAccuracy = len(model.history.accuracies) - 1

        def update_loss_dot(dot):
            prev_dot = dot.copy()
            prev_dot.move_to(lossAxis.coords_to_point(100, model.history.losses[self.countLoss]))
            dot.become(prev_dot)
            self.countLoss += 1
            if self.countLoss == len(model.history.accuracies):
                self.countLoss = len(model.history.accuracies) - 1

        accuracyDot.add_updater(update_dot)
        accuracyPath.add_updater(update_path)
        accuracyPath.set_points_as_corners([accuracyDot.get_center(), accuracyDot.get_center()])
        lossDot.add_updater(update_loss_dot)
        lossPath.add_updater(update_loss_path)
        lossPath.set_points_as_corners([lossDot.get_center(), lossDot.get_center()])
        self.add(accuracyPath, accuracyDot)
        self.add(lossPath, lossDot)

        for _ in range(150):
            for _ in range(10):
                x_batch, y_batch = spiral_data(samples=batchSize, classes=3)
                x_batch, y_batch = sklearn.utils.shuffle(x_batch, y_batch)
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
            output = image[2:-2, 2:-2]
            output = cv2.resize(output, (SIZE_TARGET, SIZE_TARGET), interpolation=cv2.INTER_NEAREST)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            output = np.array(output * 255, np.uint8)

            cv2.imshow("image", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            newImage = ImageMobject(output)
            newImage.scale(2)
            newImage.move_to(LEFT * 2)
            self.add(imageMob)

            self.play(
                Transform(imageMob, newImage, rate_func=linear),
                run_time=0.1,
            )
