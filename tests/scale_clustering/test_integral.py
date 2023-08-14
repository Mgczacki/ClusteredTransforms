import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import floats

from clustered_transforms.scale_clustering import ScaleClusterTransformer


@given(
    X=arrays(
        shape=array_shapes(min_dims=1, max_dims=1),
        unique=True,
        elements={"min_value": 0, "max_value": (2.0 - 2**-23) * 2**127},
        dtype=float,
    ),
    image_lower_cap=floats(
        allow_nan=False, allow_infinity=False, width=32, min_value=-1e10, max_value=1e10
    ),
    delta=floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=1e-4,
        exclude_min=True,
        max_value=1e10,
    ),
)
def test_normal_use_case(X, image_lower_cap, delta):
    image_upper_cap = image_lower_cap + delta

    # Create a ScaleClusterTransformer instance
    scaler = ScaleClusterTransformer(
        image_lower_cap=image_lower_cap,
        image_upper_cap=image_upper_cap,
        negative_strategy=None,
    )

    # Fit and transform the scaler on the generated data
    transformed_X = scaler.fit_transform(X)

    inv_transformed_X = scaler.inverse_transform(transformed_X)
    # Test: The inverse transform of the transform is approximately the identity
    assert np.allclose(X, inv_transformed_X, rtol=1e-5)

    # Test: Strictly monotonous increasing.
    assert np.all(np.diff(scaler.transform(np.sort(X))) >= 0)

    # Test: All values are below the upper image cap.
    assert np.all(transformed_X <= image_upper_cap)

    # Test: All values are above the lower image cap.
    assert np.all(transformed_X >= image_lower_cap)


@given(
    X=arrays(
        shape=array_shapes(min_dims=1, max_dims=1),
        unique=True,
        elements={
            "min_value": -(2.0 - 2**-23) * 2**127,
            "max_value": (2.0 - 2**-23) * 2**127,
        },
        dtype=float,
    ),
    image_lower_cap=floats(
        allow_nan=False, allow_infinity=False, width=32, min_value=-1e10, max_value=1e10
    ),
    delta=floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=1e-4,
        exclude_min=True,
        max_value=1e10,
    ),
)
def test_zero_negatives(X, image_lower_cap, delta):
    image_upper_cap = image_lower_cap + delta

    # Create a StandardScaler instance
    scaler = ScaleClusterTransformer(
        image_lower_cap=image_lower_cap,
        image_upper_cap=image_upper_cap,
        negative_strategy="zero",
    )

    # Fit and transform the scaler on the generated data
    transformed_X = scaler.fit_transform(X)

    inv_transformed_X = scaler.inverse_transform(transformed_X)

    negative_mask = X < 0

    # Test: Negatives are getting clipped to zero.
    assert np.allclose(inv_transformed_X[negative_mask], 0, rtol=1e-5)

    # Test: Strictly monotonous increasing.
    assert np.all(np.diff(scaler.transform(np.sort(X))) >= 0)

    # Test: The inverse transform of the transform is approximately the identity
    assert np.allclose(X[~negative_mask], inv_transformed_X[~negative_mask], rtol=1e-5)

    # Test: All values are below the upper image cap.
    assert np.all(transformed_X <= image_upper_cap)

    # Test: All values are above the lower image cap.
    assert np.all(transformed_X >= image_lower_cap)


if __name__ == "__main__":
    test_normal_use_case()
    test_zero_negatives()
