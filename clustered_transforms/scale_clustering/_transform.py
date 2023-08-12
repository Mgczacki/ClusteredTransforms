import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MeanShift

from ._cluster import Cluster
from ._functions import (
    inv_logarithmic_interpolation,
    inv_scaled_logistic,
    logarithmic_interpolation,
    scaled_logistic,
)


class ScaleClusterTransformer(BaseEstimator, TransformerMixin):
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

    def fit(self, X: np.array, y=None):
        self._generate_clusters(X)
        self._init_ranges()

        return self

    def _generate_clusters(self, points):
        precision = self.precision
        cluster_orders_of_magnitude = self.cluster_orders_of_magnitude

        data = np.sort(points)
        points = data.reshape(-1, 1)  # Reshape to 2D array as sklearn expects
        points = np.log10(points + precision)

        # Applying Mean Shift
        ms = MeanShift(bandwidth=cluster_orders_of_magnitude / 2)
        ms.fit(points)

        self.clusters = []

        for i, _ in enumerate(ms.cluster_centers_):
            cluster_data = data[ms.labels_ == i]
            self.clusters.append(Cluster(cluster_data))

        self.clusters = sorted(self.clusters, key=lambda p: p.mean)

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
                "Invalid uncertainty configuration."
                "Tails leave no space for spatial distribution."
            )

        self.image_range = image_range

        cluster_weights = np.array([c.w for c in self.clusters])
        total_cluster_weight = np.sum(cluster_weights)

        cumulative_weight = 0

        inter_cluster_weights = np.maximum(cluster_weights[:-1], cluster_weights[1:])
        inter_cluster_weights = inter_cluster_weights / np.sum(inter_cluster_weights)
        inter_cluster_weights = np.concatenate([[0], inter_cluster_weights])

        left_area = image_range * self.left_tail_uncertainty

        for i, c in enumerate(self.clusters):
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

    def f(self, val):
        if np.isnan(val):
            return val

        val = val + self.eps

        last_c = None

        for c in self.clusters:
            if val < c.min:
                if last_c is None:
                    # Left tail
                    return scaled_logistic(
                        val,
                        lower=self.image_lower_cap,
                        upper=self.image_lower_cap
                        + 2 * (c.y_min - self.image_lower_cap),
                        a=np.log(3) / (self.tail_midpoint_ratio * c.min),
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
            a=np.log(3) / (self.tail_midpoint_ratio * last_c.max),
            x0=last_c.max,
        )

    def inv(self, val):
        if np.isnan(val):
            return val

        last_c = None

        for c in self.clusters:
            if val < c.y_min:
                if last_c is None:
                    # Left tail
                    return inv_scaled_logistic(
                        val,
                        lower=self.image_lower_cap,
                        upper=self.image_lower_cap
                        + 2 * (c.y_min - self.image_lower_cap),
                        a=np.log(3) / (self.tail_midpoint_ratio * c.min),
                        x0=c.min,
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
        return inv_scaled_logistic(
            val,
            lower=self.image_upper_cap - 2 * (self.image_upper_cap - last_c.y_max),
            upper=self.image_upper_cap,
            a=np.log(3) / (self.tail_midpoint_ratio * last_c.max),
            x0=last_c.max,
        )
