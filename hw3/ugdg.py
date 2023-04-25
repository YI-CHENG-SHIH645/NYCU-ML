import numpy as np


def gaussian_gen(mean: float, var: float):
    rng = np.random.default_rng(None)
    uv = rng.uniform(0, 1, (1, 2))
    u, v = uv[:, 0], uv[:, 1]
    samples = mean + np.sqrt(var) * np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
    # print(f"μ = {np.mean(samples):.6f}")
    # print(f"σ = {np.std(samples):.6f}")
    return samples.item()


if __name__ == '__main__':
    print(f"A data point = {gaussian_gen(0, 1)}")
