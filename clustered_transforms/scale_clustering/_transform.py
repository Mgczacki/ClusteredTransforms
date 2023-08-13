from typing import Any, List

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
    """

    def __init__(
        self,
        left_cap=None,
        right_cap=None,
        left_tail_uncertainty=0.05,
        right_tail_uncertainty=0.05,
        inter_cluster_uncertainty=0.2,
        image_lower_cap=0,
        image_upper_cap=1,
        precision=1e-3,
        cluster_orders_of_magnitude=1,
        tail_midpoint_ratio=0.5,
        eps=1e-9,
        negative_strategy="zero",
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

    def _generate_clusters(self, points, negative=False):
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

    def _f(self, val):
        if np.isnan(val):
            return val

        val = val + self.eps

        last_c = None

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

    def _inv(self, val):
        if np.isnan(val):
            return val

        last_c = None

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
