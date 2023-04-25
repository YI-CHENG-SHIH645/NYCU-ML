# poly basis linear model data generator
import numpy as np
from ugdg import gaussian_gen


def poly_gen(n: int, a: float, w: list):
    assert n == len(w)
    rng = np.random.default_rng(None)
    x = rng.uniform(-1.0, 1.0, 1).item()
    phi_x = np.array([x ** i for i in range(1, n + 1)])
    # rng42 = np.random.default_rng(42)
    # w = rng42.uniform(-10, 10, (n, 1))
    w = np.array(w)
    e = gaussian_gen(0, a)
    return np.expand_dims(phi_x, axis=0), (w.T @ phi_x + e).item()


if __name__ == '__main__':
    print(f"A data point (x, y) = {poly_gen(n=4, a=1, w=[1, 2, 3, 4])}")
