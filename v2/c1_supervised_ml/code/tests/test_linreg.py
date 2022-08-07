import pytest
import numpy as np

from src.linreg import compute_cost, compute_gradient


@pytest.fixture
def x_train():
    return np.array([1.0, 2.0])


@pytest.fixture
def y_train():
    return np.array([300.0, 500.0])


def test_compute_cost(x_train, y_train):
    assert compute_cost(x_train, y_train, 200, 100) == 0.0


def test_compute_gradient(x_train, y_train):
    new_w, new_b = compute_gradient(x_train, y_train, 200, 100)

    assert new_w == 0.0
    assert new_b == 0.0
