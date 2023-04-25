# bayesian linear regression
import sys
import numpy as np
import matplotlib.pyplot as plt
from pblmdg import poly_gen


# b: w precision
# a: noise(ε) precision
def bayesian_lr(b: float, n: int, a: float, w: list, N=100):
    phi_mat = np.empty(shape=(0, n))
    t = np.empty(shape=(0, 1))
    for i in range(N):
        phi_x, y = poly_gen(n, a ** (-1), w)
        phi_mat = np.concatenate([phi_mat, phi_x], axis=0)
        t = np.append(t, [[y]], axis=0)

        # update prior p(w)
        SN_inv = b * np.identity(n) + a * phi_mat.T @ phi_mat
        SN = np.linalg.inv(SN_inv)
        mN = a * SN @ phi_mat.T @ t

        print(f"Add data point {(phi_x[0, 0], y)}")

        print("\nPosterior mean:")
        for wi in mN:
            print(f"{wi.item():15.10f}")

        print("\nPosterior variance:")
        for row in SN:
            for v in row:
                print(f"{v.item():15.10f}", end=",")
            sys.stdout.write("\b \b")
            print()

        print(f"\nPredictive distribution ~ N({(mN.T @ phi_x.T).item():.10f},"
              f" {(phi_x @ SN @ phi_x.T).item():.10f})")

        if i + 1 in [10, N]:
            plt.scatter(x=phi_mat[:, 0], y=t[:, 0])
            x_ = np.linspace(-1.2, 1.2, 100)
            all_points = np.expand_dims(x_, axis=1).repeat(n, axis=1)
            for z in range(1, n + 1):
                all_points[:, z - 1] **= z
            y_ = (mN.T @ all_points.T)[0]
            plt.plot(x_, y_, c='black')
            std_ = 1 / a + np.einsum('ij,jk,ki->i', all_points, SN, all_points.T)
            plt.plot(x_, y_ + std_, c='red')
            plt.plot(x_, y_ - std_, c='red')
            plt.title(f"N = {i + 1}")
            plt.show()
        print("-" * 68)


if __name__ == '__main__':
    bayesian_lr(b=0.1, n=3, a=5, w=[1, 2, 3], N=50)
