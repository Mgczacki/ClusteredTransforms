from typing import Any, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MeanShift
from typing_extensions import Self

from ._cluster import Cluster
from ._functions import (
    inv_logarithmic_interpolation,
    inv_scaled_logistic,
    logarithmic_interpolation,
    scaled_logistic,
)


class ScaleClusterTransformer(BaseEstimator, TransformerMixin):
    """A transformer that identifies scale-aware clusters of data, and maps them to a
    bounded projection space based on their importance/weight.

    Parameters
    ----------
    left_cap: float, default = None
        The minimum `x` value that will be provided.
        It sets a hard limit so that there is no uncertainty
        or tails on the left of the projection.
    right_cap: float, default = None
        The maximum `x` value that will be provided.
        It sets a hard limit so that there is no uncertainty
        or tails on the right of the projection.
    left_tail_uncertainty: float, default = 0.05
        The portion in [0,1) of projective space to leave
        for the left tail. [1]
    right_tail_uncertainty: float, default = 0.05
        The portion in [0,1) of projective space to leave
        for the right tail. [1]
    inter_cluster_uncertainty: float, default = 0.33
        The portion in [0,1) of projective space (left after tails)
        to divide in the portions between clusters.
    image_lower_cap: float, default = 0
        The asymptotic lower value of the projective space.
    image_upper_cap: float, default = 1
        The asymptotic upper value of the projective space.
    precision: float, default = 1e-3
        The minimum scale to detect.
    cluster_orders_of_magnitude: float, default = 0.75
        Cluster data that lies at most at `cluster_orders_of_magnitude`.
        Bigger value makes bigger clusters.
    tail_midpoint_ratio: float, default = 0.5
        How much of a relative increase (from the last known point)
        should pass before each tail reaches the midpoint of its alloted image.
    eps: float, default = 1e-9
        Numeric stability constant.
    negative_strategy: str, default = "zero"
        The strategy to treat negatives:
        - None | "disallow" raises an exception when trying to transform them.
        - "zero" zeroes out any negatives.
        - "mirror" builds a mirror projection (currently unimplemented).

    Notes
    -----
    [1] - The tails serve a dual purpose. Firstly, they allow for a full transformation
          of the entire input space, leaving up the possibility for predictions that lie
          outside of the known ranges of data. Even if such predictions would not be
          useful, the tails also serve a similar purpose to label smoothing in logistic
          representations of boolean variables.
    """

    def __init__(
        self,
        left_cap: Optional[float] = None,
        right_cap: Optional[float] = None,
        left_tail_uncertainty: float = 0.05,
        right_tail_uncertainty: float = 0.05,
        inter_cluster_uncertainty: float = 0.33,
        image_lower_cap: float = 0,
        image_upper_cap: float = 1,
        precision: float = 1e-3,
        cluster_orders_of_magnitude: float = 0.75,
        tail_midpoint_ratio: float = 0.5,
        eps: float = 1e-9,
        negative_strategy: str = "zero",
    ) -> None:
        self.left_cap = left_cap
        self.right_cap = right_cap
        self.left_tail_uncertainty = left_tail_uncertainty
        self.right_tail_uncertainty = right_tail_uncertainty
        self.inter_cluster_uncertainty = inter_cluster_uncertainty
        self.image_lower_cap = image_lower_cap
        self.image_upper_cap = image_upper_cap
        self.precision = precision
        self.cluster_orders_of_magnitude = cluster_orders_of_magnitude
        self.tail_midpoint_ratio = tail_midpoint_ratio
        self.eps = eps
        self.negative_strategy = negative_strategy

    def fit(self, X: np.ndarray, y: Any = None) -> Self:
        """Fit function.

        Parameters
        ----------
        X: np.ndarray
            The data to fit to.
        y: Any, default = None
            Unused. Kept for compatibility.
        """
        self.clusters_: List[Cluster] = []

        negative_mask = X < 0

        if np.any(negative_mask):
            if self.negative_strategy in (None, "disallow"):
                raise ValueError(
                    "Dataset contains negative values but the "
                    f"negative_strategy is {self.negative_strategy}."
                )
            elif self.negative_strategy == "zero":
                X = np.copy(X)
                X[negative_mask] = 0
            elif self.negative_strategy == "mirror":
                raise NotImplementedError("Mirror strategy not supported yet.")
                X_neg = X[negative_mask]
                X = X[~negative_mask]
                self._generate_clusters(X_neg, negative=True)
            else:
                raise ValueError(f"Unknown negative_strategy: {self.negative_strategy}")

        self._generate_clusters(X)
        self._init_ranges()

        return self

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        """Fit and then transform.

        Parameters
        ----------
        X: np.ndarray
            The data to fit to.
        y: Any, default = None
            Unused. Kept for compatibility.
        """
        self.fit(X)
        return self.transform(X)

    def transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        """Transform data.

        Parameters
        ----------
        X: np.ndarray
            The data to transform.
        y: Any, default = None
            Unused. Kept for compatibility.
        """
        negative_mask = X < 0

        if np.any(negative_mask):
            if self.negative_strategy in (None, "disallow"):
                raise ValueError(
                    "Dataset contains negative values but the "
                    f"negative_strategy is {self.negative_strategy}."
                )
            elif self.negative_strategy == "zero":
                X = np.copy(X)
                X[negative_mask] = 0
            elif self.negative_strategy == "mirror":
                raise NotImplementedError("Mirror strategy not supported yet.")
            else:
                raise ValueError(f"Unknown negative_strategy: {self.negative_strategy}")

        return np.vectorize(self._f)(X)

    def inverse_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        """Inverse transform.

        Parameters
        ----------
        X: np.ndarray
            The data to inverse transform.
        y: Any, default = None
            Unused. Kept for compatibility.
        """
        return np.vectorize(self._inv)(X)

    def _generate_clusters(self, points: np.ndarray, negative: bool = False):
        precision = self.precision
        cluster_orders_of_magnitude = self.cluster_orders_of_magnitude

        data = np.sort(np.abs(points))
        points = data.reshape(-1, 1)  # Reshape to 2D array as sklearn expects
        points = np.log10(points + precision)

        # Applying Mean Shift
        ms = MeanShift(bandwidth=cluster_orders_of_magnitude / 2)
        ms.fit(points)

        clusters = []

        for i, _ in enumerate(ms.cluster_centers_):
            cluster_data = data[ms.labels_ == i]
            clusters.append(Cluster(cluster_data, negative=negative))

        clusters = sorted(clusters, key=lambda p: p.mean)

        self.clusters_ = self.clusters_ + clusters

    def _init_ranges(self):
        image_range = self.image_upper_cap - self.image_lower_cap

        if self.left_cap is None:
            image_range = image_range - image_range * self.left_tail_uncertainty
        else:
            self.left_tail_uncertainty = 0.0
        if self.right_cap is None:
            image_range = image_range - image_range * self.right_tail_uncertainty
        else:
            self.right_tail_uncertainty = 0.0

        if image_range <= 0:
            raise ValueError(
                "Invalid uncertainty configuration. "
                "Tails leave no space for spatial distribution."
            )

        self.image_range = image_range

        cluster_weights = np.array([c.w for c in self.clusters_])
        total_cluster_weight = np.sum(cluster_weights)

        cumulative_weight = 0

        inter_cluster_weights = np.maximum(cluster_weights[:-1], cluster_weights[1:])
        inter_cluster_weights = inter_cluster_weights / np.sum(inter_cluster_weights)
        inter_cluster_weights = np.concatenate([[0], inter_cluster_weights])

        left_area = image_range * self.left_tail_uncertainty

        for i, c in enumerate(self.clusters_):
            c.n_w = c.w / total_cluster_weight
            if c.mass > 1:
                c_displacement = (
                    image_range * c.n_w * (1 - self.inter_cluster_uncertainty)
                )
            else:
                c_displacement = 0
            c_inter = (
                image_range * inter_cluster_weights[i] * self.inter_cluster_uncertainty
            )
            c.y_min = self.image_lower_cap + left_area + cumulative_weight + c_inter
            c.y = c.y_min + c_displacement / 2
            c.y_max = c.y_min + c_displacement
            cumulative_weight = cumulative_weight + c_displacement + c_inter

    def _f(self, val: float):
        if np.isposinf(val):
            return self.image_upper_cap

        if np.isneginf(val):
            return self.image_lower_cap

        if self.left_cap is not None and self.left_cap > val:
            raise ValueError("Provided value is below left cap.")

        if self.right_cap is not None and self.right_cap < val:
            raise ValueError("Provided value is above right cap.")

        if np.isnan(val):
            return val

        val = val + self.eps

        last_c: Any = None

        for c in self.clusters_:
            if val < c.min:
                if last_c is None:
                    # Left tail
                    return scaled_logistic(
                        val,
                        lower=self.image_lower_cap,
                        upper=self.image_lower_cap
                        + 2 * (c.y_min - self.image_lower_cap),
                        a=np.log(3) / ((self.tail_midpoint_ratio * c.min) + self.eps),
                        x0=c.min,
                    )
                else:
                    # Between clusters
                    return logarithmic_interpolation(
                        val,
                        last_c.max + self.eps,
                        last_c.y_max,
                        c.min + self.eps,
                        c.y_min,
                    )
            elif val <= c.max:
                # Within cluster
                return c.f(val)

            last_c = c

        # Right tail
        return scaled_logistic(
            val,
            lower=self.image_upper_cap - 2 * (self.image_upper_cap - last_c.y_max),
            upper=self.image_upper_cap,
            a=np.log(3) / ((self.tail_midpoint_ratio * c.min) + self.eps),
            x0=last_c.max,
        )

    def _inv(self, val: float):
        if self.image_lower_cap > val:
            raise ValueError("Provided value is below image lower cap.")

        if self.image_upper_cap < val:
            raise ValueError("Provided value is above image upper cap.")

        if np.isnan(val):
            return val

        last_c: Any = None

        for c in self.clusters_:
            if val < c.y_min:
                if last_c is None:
                    # Left tail
                    return (
                        inv_scaled_logistic(
                            val,
                            lower=self.image_lower_cap,
                            upper=self.image_lower_cap
                            + 2 * (c.y_min - self.image_lower_cap),
                            a=np.log(3)
                            / ((self.tail_midpoint_ratio * c.min) + self.eps),
                            x0=c.min,
                        )
                        - self.eps
                    )
                else:
                    # Between clusters
                    return inv_logarithmic_interpolation(
                        val,
                        last_c.max + self.eps,
                        last_c.y_max,
                        c.min + self.eps,
                        c.y_min,
                    )
            elif val <= c.y_max:
                # Within cluster
                return c.inv(val)

            last_c = c

        # Right tail
        return (
            inv_scaled_logistic(
                val,
                lower=self.image_upper_cap - 2 * (self.image_upper_cap - last_c.y_max),
                upper=self.image_upper_cap,
                a=np.log(3) / ((self.tail_midpoint_ratio * c.min) + self.eps),
                x0=last_c.max,
            )
            - self.eps
        )
