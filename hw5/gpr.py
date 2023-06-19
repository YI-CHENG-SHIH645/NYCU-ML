import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def rational_quad_kernel(X, a, b):
    # return a covariance matrix C_N
    # where C_N(x_n, x_m) = k(x_n, x_m) + 1/beta (if diag)
    # so the k here is the kernel function : rational quadratic kernel

    # |d|^2 (squared euclidean) between every pair of points, form a matrix
    dists = squareform(pdist(X, metric='sqeuclidean'))
    tmp = dists / (2 * a * b ** 2)
    base = (1 + tmp)
    K = base ** (-a)

    # kernel(x, x) = 1, where x = x
    np.fill_diagonal(K, 1)

    # noise parameter beta = 5
    K[np.diag_indices_from(K)] += 5 ** -1

    return K, dists, base


def predict(x, X, a, b, alpha, lower):
    # x shape : (2400, 1), X shape : (34, 1)
    # dist_mat = (2400, 34)
    dist_mat = cdist(x.reshape(-1, 1), X, metric="sqeuclidean")
    K = (1 + dist_mat / (2 * a * b ** 2)) ** (-a)

    # kT CN**-1 t
    y_mean = K.dot(alpha)

    V = cho_solve((lower, True), K.T)

    # c - kT CN**-1 kT
    y_var = np.full(shape=(x.shape[0], ), fill_value=1+5**-1) \
        - np.einsum("ij,ji->i", K, V)
    y_std = np.sqrt(y_var)

    return y_mean, y_std


def obj_func(theta, X, y):
    a, b = theta
    K, dists, base = rational_quad_kernel(X, a, b)

    L = cholesky(K, lower=True)
    al = cho_solve((L, True), y[:, np.newaxis])

    # second term
    log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y[:, np.newaxis], al)
    # first term
    log_likelihood_dims -= np.log(np.diag(L)).sum()
    # third term
    log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
    log_likelihood = log_likelihood_dims.sum(-1)

    # partial C_N over theta_1
    a_gradient = K * (-a * np.log(base) + dists / (2 * b ** 2 * base))
    a_gradient = a_gradient[:, :, np.newaxis]

    # partial C_N over theta_2
    b_gradient = dists * K / (b ** 2 * base)
    b_gradient = b_gradient[:, :, np.newaxis]

    K_gradient = np.dstack((a_gradient, b_gradient))

    # second term
    # the r.h.s. is t^T @ C_N^{-1} @ C_N^{-1} @ t
    tmp = np.einsum("ik,jk->ijk", al, al)
    # first term
    # the r.h.s. is C_N^{-1}
    tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]

    # calculate the gradient respectively
    log_likelihood_gradient_dims = 0.5 * np.einsum("ijl,jik->kl", tmp, K_gradient)
    log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

    return -log_likelihood, -log_likelihood_gradient  # log_likelihood_gradient : (2, )


def plot_gpr_samples(x_val, y_mu, y_sigma, ax):
    ax.plot(x_val, y_mu, color="black", label="Mean")
    ax.fill_between(
        x_val,
        y_mu - y_sigma,
        y_mu + y_sigma,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def main():
    data = np.loadtxt("data/input.data")
    X, y = data[:, [0]], data[:, 1]

    a, b = 1, 1
    # Rational Quadratic
    K, _, _ = rational_quad_kernel(X, a, b)
    lower = cholesky(K, lower=True)
    alpha = cho_solve((lower, True), y)

    x = np.linspace(-60, 60, 2400)

    y_mean, y_std = predict(x, X, a, b, alpha, lower)

    _, ax1 = plt.subplots()
    plot_gpr_samples(x, y_mean, y_std, ax1)
    ax1.scatter(X[:, 0], y, color="red", zorder=10, label="Observations")
    ax1.legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
    ax1.set_title("Without Optimizing the kernel parameters (a, b)")

    print(f"Without Optimizing, MSE: {np.sqrt(((predict(X, X, a, b, alpha, lower)[0] - y)**2).mean()):4f}")

    init_theta = np.array([1., 1.])
    res = minimize(obj_func, init_theta, args=(X, y), method="L-BFGS-B",
                   jac=True, bounds=np.array([[1e-5, 1e5], [1e-5, 1e5]]))
    a, b = res.x
    print(f"a = {a:.4f} \nb = {b:.4f}")
    K, _, _ = rational_quad_kernel(X, a, b)
    lower = cholesky(K, lower=True)
    alpha = cho_solve((lower, True), y)

    y_mean, y_std = predict(x, X, a, b, alpha, lower)

    _, ax2 = plt.subplots()
    plot_gpr_samples(x, y_mean, y_std, ax2)
    ax2.scatter(X[:, 0], y, color="red", zorder=10, label="Observations")
    ax2.legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
    ax2.set_title("Optimize the kernel parameters (a, b)")

    print(f"Optimized, MSE: {np.sqrt(((predict(X, X, a, b, alpha, lower)[0] - y) ** 2).mean()): .4f}")

    # plt.show()


if __name__ == '__main__':
    main()
