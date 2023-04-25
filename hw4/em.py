import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import accuracy_score, confusion_matrix


def display_metrics(y_true, y_pred):
    cf = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cf.ravel()
    print("confusion matrix:\n", pd.DataFrame(cf,
                                              columns=["Predict cluster 1", "Predict cluster 2"],
                                              index=["Is cluster 1", "Is cluster 2"]))
    print(f"Sensitivity (recall) : {tp / (tp + fn)}")
    print(f"Specificity (TNR) : {tn / (tn + fp)}")


def AInstKClusDBernoulli(X, MU):
    return np.prod(
        MU[None, :, :] ** X[:, None, :] * (1-MU[None, :, :]) ** (1-X[:, None, :]),
        axis=2
    )


def E_LL_MB(X, W, MU, r):
    return np.sum(r * (np.log(W)[None, :] + np.sum(np.log((
        MU[None, :, :] ** X[:, None, :] * (1-MU[None, :, :]) ** (1-X[:, None, :])
        .clip(min=1e-50)
    )), axis=2)), axis=(0, 1))


if __name__ == '__main__':
    image_size = 28
    with open("train-images-idx3-ubyte", "rb") as f_img, \
            open("train-labels-idx1-ubyte", "rb") as f_lbl:
        f_img.read(16)
        img = (
            np.frombuffer(f_img.read(image_size * image_size * 60000), dtype=np.uint8)
            .astype(np.float32)
            .reshape((60000, image_size * image_size))
            // 128
        )
        f_lbl.read(8)
        lbl = (
            np.frombuffer(f_lbl.read(60000), dtype=np.uint8)
            .astype(np.int64)
        )

    # weights for each cluster
    weights = np.full((10, ), 0.1)
    weights /= np.sum(weights)

    # for every x_i, it is a mixture of bernoulli
    mu = np.random.rand(10, 784)
    mu /= mu.sum(axis=0, keepdims=True)

    e_ll_mb_old = None
    w_likelihood_norm = np.empty(shape=(60000, 10))

    for i in range(1, 1001):
        print(f"Epoch:{i}, ", end='')
        # ------------------------ E step ------------------------
        # P(X=Cm|mu) : (60000, 10)
        likelihood = AInstKClusDBernoulli(img, mu)

        w_likelihood = weights[None, :] * likelihood

        w_likelihood = w_likelihood.clip(min=np.min(w_likelihood[np.nonzero(w_likelihood)]))
        # resp : (60000, 10)
        w_likelihood_norm = np.exp(np.log(w_likelihood) - np.log(w_likelihood.sum(1, keepdims=True)))
        # ------------------------ E step ------------------------

        # ------------------------ M step ------------------------
        weights = w_likelihood_norm.mean(axis=0)
        mu = w_likelihood_norm.T @ img / w_likelihood_norm.sum(axis=0)[:, None]
        # ------------------------ M step ------------------------

        if i == 1:
            e_ll_mb_old = E_LL_MB(img, weights, mu, w_likelihood_norm)
            print()
        else:
            e_ll_mb = E_LL_MB(img, weights, mu, w_likelihood_norm)
            diff = abs(e_ll_mb-e_ll_mb_old)
            print(f"Difference : {diff:.0f}")
            if diff < 1e-5:
                print(f"****** Early Stopping at {i} ****** ")
                break
            e_ll_mb_old = e_ll_mb
    print()
    mapping = {}
    for gt in range(10):
        mapping.update({gt: mode(np.argmax(w_likelihood_norm[lbl == gt], axis=1), keepdims=False)[0]})
    for num in range(10):
        print(f"for number: {num}")
        pred = np.where(np.argmax(w_likelihood_norm[:, list(mapping.values())], axis=1) == num, 0, 1)
        y = np.where(lbl == num, 0, 1)
        display_metrics(y, pred)
        print()

    print(f"final acc: {accuracy_score(lbl, np.argmax(w_likelihood_norm[:, list(mapping.values())], axis=1))}")
