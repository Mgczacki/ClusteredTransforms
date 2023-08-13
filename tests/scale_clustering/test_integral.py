import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays

from clustered_transforms.scale_clustering import ScaleClusterTransformer


@given(
    X=arrays(
        shape=array_shapes(min_dims=1, max_dims=1),
        unique=True,
        elements={"min_value": 0, "max_value": (2.0 - 2**-23) * 2**127},
        dtype=float,
    )
)
def test_normal_use_case(X):
    image_upper_cap = 2
    image_lower_cap = -4

    # Create a StandardScaler instance
    scaler = ScaleClusterTransformer(
        image_lower_cap=image_lower_cap, image_upper_cap=image_upper_cap
    )

    # Fit and transform the scaler on the generated data
    transformed_X = scaler.fit_transform(X)

    # Test: The inverse transform of the transform is approximately the identity
    inv_transformed_X = scaler.inverse_transform(transformed_X)

    assert np.allclose(X, inv_transformed_X, rtol=1e-5, atol=1e-5)

    # Test: All values are below x
    assert np.all(transformed_X <= image_upper_cap)

    # Test: All values are above y
    assert np.all(transformed_X >= image_lower_cap)


if __name__ == "__main__":
    test_normal_use_case()
