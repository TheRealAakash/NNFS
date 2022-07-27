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
model.add(LayerDropout(0.1))
model.add(LayerDense(128, 256, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(ActivationReLU())
model.add(LayerDropout(0.1))
model.add(LayerDense(256, 3))
model.add(ActivationSoftmax())
model.set(loss=LossCategoricalCrossentropy(), optimizer=OptimizerAdam(learning_rate=0.05, decay=5e-7),
          accuracy=AccuracyCategorical())
model.finalize()

SIZE = 300
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
        lossGraph = Axes(
            y_range=[0, 3, 1],
            x_range=[0, 100, 10],
            y_axis_config={
                "include_numbers": True,
            },
            axis_config={
                "include_tip": False,
            }
        )
        # rect behind the loss graph

        lossGraph.scale(0.25)
        lossGraph.move_to(RIGHT * 4 + DOWN * 1)
        lossTitle = Text("Loss", font_size=15)
        rect = Rectangle(
            width=lossGraph.x_axis.get_width(),
            height=lossGraph.y_axis.get_height(),
            stroke_width=0,
            fill_color="#2f2f2f",
            fill_opacity=0.5,
        )
        rect.move_to(lossGraph.x_axis.get_left(), LEFT).shift(UP * 0.75)
        lossTitle.move_to(rect.get_center() + UP * 0.5)
        self.add(rect)
        self.add(lossGraph)
        self.add(lossTitle)

        accuracyGraph = Axes(
            y_range=[0, 1, 0.2],
            x_range=[0, 100, 10],
            axis_config={
                "include_tip": False,
            }

        )

        accuracyGraph.scale(0.25)
        accuracyGraph.move_to(RIGHT * 4 + UP * 1)
        rect2 = Rectangle(
            width=accuracyGraph.x_axis.get_width(),
            height=accuracyGraph.y_axis.get_height(),
            stroke_width=0,
            fill_color="#2f2f2f",
            fill_opacity=0.5,
        )
        rect2.move_to(accuracyGraph.x_axis.get_left(), LEFT).shift(UP * 0.75)
        accTitle = Text("Accuracy", font_size=15)
        accTitle.move_to(rect2.get_center() + UP * 0.5)
        self.add(rect2)
        self.add(accuracyGraph)
        self.add(accTitle)

        loss = []
        xs = [0]
        accuracy = [0]
        graph = lossGraph.plot_line_graph(x_values=[0], y_values=[0])
        graph2 = accuracyGraph.plot_line_graph(x_values=[0], y_values=[0])
        batchSize = 16
        for _ in range(150):
            x_batch, y_batch = spiral_data(samples=batchSize, classes=3)
            x_batch, y_batch = sklearn.utils.shuffle(x_batch, y_batch)
            ident = np.eye(3)
            ident[2][0] = 1
            ident[2][1] = 1
            ident[2][2] = 0
            y_col_b = ident[y_batch]
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
            image = cv2.resize(image, (SIZE_TARGET, SIZE_TARGET), interpolation=cv2.INTER_NEAREST)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image2 = np.array(image * 255, np.uint8)

            cv2.imshow("image", cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            newImage = ImageMobject(image2)
            newImage.scale(2)
            newImage.move_to(LEFT * 2)
            loss.append(model.lossScore)
            accuracy.append(model.accuracyScore)
            xs.append(len(loss))
            # lossGraph.x_axis.x_range = [0, len(loss) + 1]
            # accuracyGraph.x_axis.x_range = [0, len(accuracy) + 1]
            self.add(imageMob)

            newLossGraph = Axes(
                y_range=[0, 3, 1],
                x_range=[0, len(loss) + 1, max(1, len(loss) // 3)],
                y_axis_config={
                    "include_numbers": True,
                    "font_size": 50,
                },
                axis_config={
                    "include_tip": False,
                }
            )
            # rect behind the loss graph
            newLossGraph.scale(0.25)
            newLossGraph.move_to(RIGHT * 4 + DOWN * 1)

            newaccuracyGraph = Axes(
                y_range=[0, 1, 0.2],
                x_range=[0, len(loss) + 1, max(1, len(loss) // 5)],
                y_axis_config={
                    "include_numbers": True,
                    "font_size": 50,
                },
                axis_config={
                    "include_tip": False,
                }
            )

            newaccuracyGraph.scale(0.25)
            newaccuracyGraph.move_to(RIGHT * 4 + UP * 1)

            newgraph = newLossGraph.plot_line_graph(x_values=xs, y_values=loss, add_vertex_dots=False,
                                                    line_color=ORANGE)

            newaccgraph = newaccuracyGraph.plot_line_graph(x_values=xs, y_values=accuracy, add_vertex_dots=False,
                                                           line_color=GREEN)

            # update axes
            self.play(
                ReplacementTransform(lossGraph, newLossGraph),
                ReplacementTransform(accuracyGraph, newaccuracyGraph),
                ReplacementTransform(imageMob, newImage),
                ReplacementTransform(graph, newgraph),
                ReplacementTransform(graph2, newaccgraph),
                rect2.animate.move_to(newaccuracyGraph.x_axis.get_left(), LEFT).shift(UP * 0.75),
                run_time=0.1
            )
            graph = newgraph
            graph2 = newaccgraph
            lossGraph = newLossGraph
            accuracyGraph = newaccuracyGraph
