import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


def gaussian_gen(mean: float, var: float, n: int):
    rng = np.random.default_rng(None)
    uv = rng.uniform(0, 1, (n, 2))
    u, v = uv[:, 0], uv[:, 1]
    samples = mean + np.sqrt(var) * np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
    # print(f"μ = {np.mean(samples):.6f}")
    # print(f"σ = {np.std(samples):.6f}")
    return samples


def sigmoid(x):
    return 1/(1+np.exp(-x))


def neg_loglikelihood(y, y_hat, eps=1e-10):
    return (- y * np.log(y_hat + eps) - (1 - y) * np.log(1 - y_hat + eps)).mean()


def display_metrics(name: str):
    print(f"w \n {w}")
    cf = confusion_matrix(y, sigmoid(X @ w) > 0.5)
    tn, fp, fn, tp = cf.ravel()
    print("confusion matrix:\n", pd.DataFrame(cf,
                                              columns=["Predict cluster 1", "Predict cluster 2"],
                                              index=["Is cluster 1", "Is cluster 2"]))
    print(f"Sensitivity (recall) : {tp / (tp + fn)}")
    print(f"Specificity (TNR) : {tn / (tn + fp)}")
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=(sigmoid(X @ w) > 0.5).ravel()).set(title=name)
    plt.show()


if __name__ == '__main__':
    # Case 1:
    mx1 = my1 = 1
    vx1 = vy1 = 2
    mx2 = my2 = 10
    vx2 = vy2 = 2
    n = 50
    lr = 1
    patience = 20
    cls1 = np.array(list(zip(gaussian_gen(mx1, vx1, n), gaussian_gen(my1, vy1, n))))
    cls2 = np.array(list(zip(gaussian_gen(mx2, vx2, n), gaussian_gen(my2, vy2, n))))

    # y: (2n, 1), y_hat: (2n, 1), X: (2n, f+1), w: (f+1, 1)
    X = np.concatenate([cls1, cls2], axis=0)
    X = np.concatenate([X, np.ones((2*n, 1))], axis=1)
    assert X.shape == (2*n, 3)
    y = np.concatenate([np.array([0]*n), np.array([1]*n)], axis=0)[:, None]
    assert y.shape == (2*n, 1)

    idx = list(range(n*2))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y.ravel()).set(title='Ground Truth')
    plt.show()

    print("\n\nGradient Descent")
    w = np.random.random((3, 1))
    i = 0
    acc_n_steps_before = 0
    while True:
        i += 1
        # print(f"epoch {i+1}")
        y_hat = sigmoid(X@w)
        grad_w = 1/n * X.T @ (y_hat-y)
        w = w - lr * grad_w
        loss = neg_loglikelihood(y, sigmoid(X@w))
        acc = accuracy_score(y, sigmoid(X@w) > 0.5)
        if i % patience == 0:
            if acc > acc_n_steps_before:
                acc_n_steps_before = acc
            else:
                print(f"early stopping at epoch {i}")
                break
        # print(f"neg_loglikelihood : {loss}", end=", ")
        # print(f"accuracy_score : {acc}")
    display_metrics("Gradient Descent")

    print("\n\nNewton's Method")
    w = np.random.random((3, 1))
    i = 0
    acc_n_steps_before = 0
    while True:
        i += 1
        # print(f"epoch {i + 1}")
        y_hat = sigmoid(X@w)
        hessian = X.T @ np.diag((y_hat * (1-y_hat)).ravel()) @ X
        grad_w = 1 / n * X.T @ (y_hat - y)
        if np.linalg.det(hessian) == 0:
            w = w - lr * grad_w
        else:
            w = w - np.linalg.inv(hessian) @ grad_w
        loss = neg_loglikelihood(y, sigmoid(X @ w))
        acc = accuracy_score(y, sigmoid(X @ w) > 0.5)
        if i % patience == 0:
            if acc > acc_n_steps_before:
                acc_n_steps_before = acc
            else:
                print(f"early stopping at epoch {i}")
                break
        # print(f"neg_loglikelihood : {neg_loglikelihood(y, sigmoid(X @ w))}", end=", ")
        # print(f"accuracy_score : {accuracy_score(y, sigmoid(X @ w) > 0.5)}")
    display_metrics("Newton's Method")
