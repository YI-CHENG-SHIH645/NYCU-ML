import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import accuracy_score


class MyGaussianNB:
    def __init__(self):
        self.mu = None
        self.var = None
        self.cls_prior = None

    def fit(self, x, y):
        mu_var = (
            pd.DataFrame(np.concatenate((x, np.expand_dims(y, axis=1)), axis=1),
                         columns=list(range(x.shape[1])) + ['label'])
              .astype({"label": int})
              .groupby('label')
              .agg(['mean', 'var'])
        )
        self.mu = mu_var.loc[:, pd.IndexSlice[:, "mean"]].droplevel(1, axis=1).values
        self.var = mu_var.loc[:, pd.IndexSlice[:, "var"]].droplevel(1, axis=1).values + 1000
        self.cls_prior = np.log(np.unique(y, return_counts=True)[1] / np.size(y))

    def predict(self, x):
        # jll = log(P(D|θ)), based on
        # 1. strong assumption of conditional independent
        # 2. the assumption that P(xi|θ) is gaussian distributed for each i
        # self.mu: (C=10, D=784)
        # self.var: (C=10, D=784)
        # self.cls_prior: (C=10, )
        # x: (N=10000, D=784)
        jll = -0.5 * np.sum(np.log(2.0 * np.pi * self.var), axis=1)[:, None]
        jll = jll - 0.5 * np.sum(((x[None, :, :] - self.mu[:, None, :])**2 / self.var[:, None, :]), 2)
        # log(p(θ))
        log_prior = self.cls_prior[:, None]
        log_numerator = (log_prior+jll).T
        # log(p(D)), sum -> A/Q + B/Q + C/Q + ... = 1 -> Q = A + B + C
        log_d = scipy.special.logsumexp(log_numerator, axis=1)
        posterior_prob = np.exp(log_numerator - log_d[:, None])

        return posterior_prob


class MyCategoricalNB:
    def __init__(self):
        self.smoothed_class_count = None
        self.feature_log_prob = None
        self.cls_log_prior = None

    def fit(self, x, y):
        # (784, 10, 32)
        # 所有 instance 的 feature i 分 label 計算 bin count
        self.smoothed_class_count = (
                pd.DataFrame(np.concatenate((x//8, np.expand_dims(y, axis=1)), axis=1),
                             columns=list(range(x.shape[1])) + ['label'])
                .astype({"label": int})
                .groupby(['label'])
                .apply(lambda z: z.drop('label', axis=1).apply(pd.Series.value_counts))
                .fillna(0)
                .T.values.reshape((-1, 10, 32)) + 1
        )
        # P(xi|θ)
        self.feature_log_prob = np.log(self.smoothed_class_count)-np.log(self.smoothed_class_count.sum(axis=1)[:, None])
        self.cls_log_prior = np.log(np.unique(y, return_counts=True)[1] / np.size(y))

    def predict(self, x):
        # jll = log(P(D|θ)), based on
        # strong assumption of conditional independent
        # self.smoothed_class_count: (D=784, C=10, N_BIN=32)
        # self.feature_log_prob: (D=784, C=10, N_BIN=32)
        # self.cls_log_prior: (C=10, )
        # x: (N=10000, D=784)
        x = (x//8).astype(np.int64)
        # (D=784, C=10, N_BIN=32) -> (N=10000, D=784, C=10) -> (N=10000, C=10)
        total_jll = self.feature_log_prob[np.arange(784), :, x].sum(axis=1) + self.cls_log_prior
        # It's actually doing softmax
        marginalized_jll = np.exp(total_jll - scipy.special.logsumexp(total_jll, axis=1)[:, None])
        return marginalized_jll


if __name__ == '__main__':
    image_size = 28
    with open("train-images-idx3-ubyte", "rb") as f_img, \
            open("t10k-images-idx3-ubyte", "rb") as f_img_test, \
            open("train-labels-idx1-ubyte", "rb") as f_lbl, \
            open("t10k-labels-idx1-ubyte", "rb") as f_lbl_test:
        f_img.read(16)
        f_img_test.read(16)
        img = np.concatenate((np.frombuffer(f_img.read(image_size * image_size * 60000),
                                            dtype=np.uint8).astype(np.float32)
                              .reshape((60000, image_size * image_size)),
                              np.frombuffer(f_img_test.read(image_size * image_size * 10000),
                                            dtype=np.uint8).astype(np.float32)
                              .reshape((10000, image_size * image_size))), axis=0)
        f_lbl.read(8)
        f_lbl_test.read(8)
        lbl = np.concatenate((np.frombuffer(f_lbl.read(60000),
                                            dtype=np.uint8).astype(np.int64),
                              np.frombuffer(f_lbl_test.read(10000),
                                            dtype=np.uint8).astype(np.int64)))

        # data : (70000, 784), label : (70000, )
        xy = np.concatenate((img, np.expand_dims(lbl, 1)), axis=1)
        xy_train = xy[:60000]
        xy_test = xy[60000:]
        np.random.shuffle(xy_train)
        np.random.shuffle(xy_test)
        # x_train : (60000, 784), y_train : (60000, )
        x_train, y_train = xy_train[:, :-1], xy_train[:, -1]
        # x_test : (10000, 784), y_test : (10000, )
        x_test, y_test = xy_test[:, :-1], xy_test[:, -1]

        # print(x_train.shape, y_train.shape)
        # print(x_test.shape, y_test.shape)
        np.set_printoptions(precision=3, suppress=True)
        print("Categorical Naive Bayes: ")
        m = MyCategoricalNB()
        m.fit(x_train, y_train)
        y_pred_prob = m.predict(x_test)
        print(f"predicted proba : \n {y_pred_prob[0]}")
        print(f"predicted label : {np.argmax(y_pred_prob, axis=1)[0]}")
        print(f"Gt : {int(y_test[0])}")
        print(f"accuracy: {accuracy_score(y_test, np.argmax(y_pred_prob, axis=1))}")
        print()
        print("Gaussian Naive Bayes: ")
        m = MyGaussianNB()
        m.fit(x_train, y_train)
        y_pred_prob = m.predict(x_test)
        print(f"predicted proba : \n {y_pred_prob[0]}")
        print(f"predicted label : {np.argmax(y_pred_prob, axis=1)[0]}")
        print(f"Gt : {int(y_test[0])}")
        print(f"accuracy: {accuracy_score(y_test, np.argmax(y_pred_prob, axis=1))}")
