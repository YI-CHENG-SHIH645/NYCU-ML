import numpy as np

from pblmdg import poly_gen


if __name__ == '__main__':
    b = 1
    n = 4
    a = 1
    w = [1, 2, 3, 4]
    m0 = 0
    S0 = b**(-1) * np.identity(n)
    S0_inv = np.linalg.inv(S0)
    phi_mat = np.empty(shape=(0, n))
    t = np.empty(shape=(0, 1))
    mN = 0
    for i in range(5):
        x, y = poly_gen(n, a, w)
        phi_mat = np.concatenate([phi_mat, x], axis=0)
        t = np.append(t, [[y]], axis=0)
        SN = phi_mat.T @ phi_mat
        print(SN.shape)
        mN = np.linalg.inv(SN) @ phi_mat
