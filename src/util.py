import math

import matplotlib as mpl
from manim import *


class ShrinkToPoint(Transform):
    """
    Transform to shrink an object to a specific point.
    """

    def __init__(
        self, mobject: Mobject, point: np.ndarray, **kwargs
    ) -> None:
        self.point = point
        super().__init__(mobject, introducer=True, **kwargs)

    def create_target(self) -> Mobject:
        end = super().create_starting_mobject()
        end.scale(0)
        end.move_to(self.point)
        return end

    def create_starting_mobject(self) -> Mobject:
        return self.mobject


class ArrayMobject:
    """
    A way to represent arrays in Manim.
    # TODO it doesn't actually subclass Mobject, which it should
    """

    def __init__(self, array, cmap, vmin, vmax):
        self.array = array
        self.cmap = cmap
        self.img_values = self._calculate_img_values(vmin, vmax)

    def _calculate_img_values(self, vmin, vmax):
        img_values = np.empty((1, len(self.array), 3))
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        for idx, v in enumerate(self.array):
            img_values[:, idx, :] = self.cmap(norm(v.item()))[:3]
        img_values *= 255
        img_values = img_values.astype(np.uint8)
        return img_values

    def create_mobject(self):
        # Tensor coloured squares
        img = ImageMobject(self.img_values)
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        img.height = 1

        # Labels for each square
        labels = []
        for idx, v in enumerate(self.array):
            # TODO sort out -0.0 labels
            label = MathTex("{:.1f}".format(v.item())).set_color(WHITE).scale(0.8)
            label.move_to(img.get_center()).shift(RIGHT * (idx - (len(self.array) - 1)/2))
            labels.append(label)

        # Group together and return
        group = Group()
        group.add(img, *labels)
        return group

    def create_splits(self):
        splits = []
        for idx in range(len(self.array)):
            print(self.img_values[:, idx, :])
            img = ImageMobject([self.img_values[:, idx, :]])
            img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            img.height = 1
            label = MathTex("{:.1f}".format(self.array[idx].item())).set_color(WHITE).scale(0.8)
            label.move_to(img.get_center())
            group = Group()
            group.add(img, label)
            splits.append(group)
        return splits


def calculate_angle(mobject_1, mobject_2):
    """
    Calculate the angle between two mobjects.
    """
    dx = mobject_2.get_x() - mobject_1.get_x()
    dy = mobject_2.get_y() - mobject_1.get_y()
    angle = math.atan2(dy, dx)
    return angle


def create_filter(colour):
    """
    Create a filter (funnel) object using Manim's Polygon class
    """
    position_list = [
        [-1, 1.5, 0],
        [1, 1.5, 0],
        [0.2, 0.25, 0],
        [0.2, -0.5, 0],
        [-0.2, -0.3, 0],
        [-0.2, 0.25, 0],
    ]
    rect = Polygon(*position_list, fill_opacity=1).set_fill(colour).set_stroke(BLACK)
    return rect
