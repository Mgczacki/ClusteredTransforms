import numpy as np


class Cluster:
    """A Cluster of points that supports interpolation between its min/max values.

    Parameters
    ----------
    points: list
        A list of points.
    """

    def __init__(self, points: list, negative: bool = False):
        self.points = np.sort(points)
        self.negative = negative  # If the Cluster is being used for negative values.
        m = -1 if negative else 1

        self.mean = m * np.mean(self.points)  # The mean of the points.
        self.std = np.std(self.points)  # The standard deviation of the points.
        self.mass = len(self.points)  # The number of points.
        self.min = m * np.min(self.points)  # The minimum point.
        self.max = m * np.max(self.points)  # The maximum point.
        self.w = self._get_w()  # The weight for the cluster.
        self.n_w = None  # The normalized weight for the cluster.
        self.y_max = None  # The image for the `self.max`.
        self.y_min = None  # The image for the `self.min`.

    def __repr__(self) -> str:
        return (
            f"Cluster - ["
            f"min: {self.min}, "
            f"min: {self.max}, "
            f"y_min: {self.y_min}, "
            f"y_max: {self.y_max}]"
        )

    def _get_w(self):
        """Calcualate the weight for the cluster."""
        if self.mass == 1:
            return 1
        return np.log(max(1, self.std / np.abs(self.mean)) * self.mass) + 1

    def f(self, x):
        """Interpolation function for the cluster."""

        if len(self.points) <= 1:
            return self.y_min

        ratio = (x - self.min) / (self.max - self.min)
        y = self.y_min + (self.y_max - self.y_min) * ratio
        return y

    def inv(self, y):
        """Inverse interpolation function for the cluster."""

        if len(self.points) <= 1:
            return self.min

        ratio = (y - self.y_min) / (self.y_max - self.y_min)
        x = self.min + ratio * (self.max - self.min)
        return x
