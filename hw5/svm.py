import os.path as osp
import numpy as np
from libsvm.svmutil import *
from scipy.spatial.distance import *


def parse(filename):
    with open(filename, 'r') as f:
        return np.array(
            [row.strip().split(',') for row in f.readlines()],
            dtype=np.float64
        )


def mnist():
    y_train = parse(osp.join("data", "Y_train.csv"))  # (5000, 1)
    x_train = parse(osp.join("data", "X_train.csv"))  # (5000, 784)
    y_test = parse(osp.join("data", "Y_test.csv"))  # (2500, 1)
    x_test = parse(osp.join("data", "X_test.csv"))    # (2500, 784)

    return x_train, x_test, np.squeeze(y_train), np.squeeze(y_test)


def svm_test(y_train, x_train, y_test, x_test, params):
    # -q : quiet mode (no outputs).
    params += " -q "
    params = svm_parameter(params)
    prob = svm_problem(y_train, x_train)
    model = svm_train(prob, params)

    # The return tuple contains
    #     p_labels: a list of predicted labels
    #     p_acc: a tuple including  accuracy (for classification), mean-squared
    #            error, and squared correlation coefficient (for regression).
    #     p_vals: a list of decision values or probability estimates (if '-b 1'
    #             is specified). If k is the number of classes, for decision values,
    #             each element includes results of predicting k(k-1)/2 binary-class
    #             SVMs. For probabilities, each element contains k values indicating
    #             the probability that the testing instance is in each class.
    #             Note that the order of classes here is the same as 'model.label'
    #             field in the model structure.
    _, test_acc, _ = svm_predict(y_test, x_test, model)

    return test_acc[0]


def svm_cross_validation(y_train, x_train, params: str, k_fold: int):
    # If '-v' is specified in 'options' (i.e., cross validation)
    # either accuracy (ACC) or mean-squared error (MSE) is returned
    params += f" -v {k_fold:d} -q "
    params = svm_parameter(params)
    prob = svm_problem(y_train, x_train)
    val_acc = svm_train(prob, params)

    return val_acc


def grid_search(y_train, x_train, y_test, x_test, kernel_id: int):
    cs = [10 ** i for i in range(-2, 3)]
    rs = range(-2, 3)
    # 784**-1 is the default 1/num_features
    gs = [784**-1] + [1 ** -i for i in range(3)]
    ds = range(0, 5)
    k_fold = 3

    best_val_acc = 0.0
    # -t kernel_type : set type of kernel function (default 2)
    # 	0 -- linear: u'*v
    # 	1 -- polynomial: (gamma*u'*v + coef0)^degree
    # 	2 -- radial basis function: exp(-gamma*|u-v|^2)
    # 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
    kernel_types = ["linear", "polynomial", "radial basis function", "sigmoid"]

    if kernel_types[kernel_id] == "linear":
        print("Number of combinations: ", len(cs))
        best_params = params = " -t 0 -c {} "
        for c in cs:
            # -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)

            val_acc = svm_cross_validation(y_train,
                                           x_train,
                                           params.format(c),
                                           k_fold)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = params.format(c)

    elif kernel_types[kernel_id] == "polynomial":
        print("Number of combinations: ", len(cs) * len(rs) * len(gs) * len(ds))
        best_params = params = " -t 1 -c {} -r {} -g {} -d {} "

        for c in cs:
            # -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)

            for r in rs:
                # -r coef0 : set coef0 in kernel function (default 0)

                for g in gs:
                    # -g gamma : set gamma in kernel function (default 1/num_features)

                    for d in ds:
                        # -d degree : set degree in kernel function (default 3)

                        val_acc = svm_cross_validation(y_train,
                                                       x_train,
                                                       params.format(c, r, g, d),
                                                       k_fold)

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_params = params.format(c, r, g, d)

    elif kernel_types[kernel_id] == "radial basis function":
        print("Number of combinations: ", len(cs) * len(gs))
        best_params = params = " -t 2 -c {} -g {} "

        for c in cs:

            for g in gs:

                val_acc = svm_cross_validation(y_train,
                                               x_train,
                                               params.format(c, g),
                                               k_fold)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = params.format(c, g)

    else:
        raise ValueError(f"Unknown Kernel ID : {kernel_id}")

    print("v" * 30)
    print(kernel_types[kernel_id])
    print("best params: ", best_params)
    print("best val acc on training: ", best_val_acc)
    print("test acc using best params : ")
    svm_test(y_train, x_train, y_test, x_test, best_params)
    print("^" * 30)


def rbf_kernel(u, v, gamma):
    return np.exp(gamma * cdist(u, v, 'sqeuclidean'))


def main():
    x_train, x_test, y_train, y_test = mnist()

    # -t kernel_type : set type of kernel function (default 2)
    # 	0 -- linear: u'*v
    # 	1 -- polynomial: (gamma*u'*v + coef0)^degree
    # 	2 -- radial basis function: exp(-gamma*|u-v|^2)
    # 	3 -- sigmoid: tanh(gamma*u'*v + coef0)

    part = "3"

    if part == "1":
        svm_test(y_train, x_train, y_test, x_test, " -t 0 ")
        svm_test(y_train, x_train, y_test, x_test, " -t 1 ")
        svm_test(y_train, x_train, y_test, x_test, " -t 2 ")

    elif part == "2":
        grid_search(y_train, x_train, y_test, x_test, 0)
        grid_search(y_train, x_train, y_test, x_test, 1)
        grid_search(y_train, x_train, y_test, x_test, 2)

    elif part == "3":
        # 784**-1 is the default 1/num_features
        gs = [784**-1] + [1 ** -i for i in range(3)]
        cs = [10 ** i for i in range(-2, 3)]

        n_train = x_train.shape[0]

        best_val_acc = 0.0
        best_params = None
        best_g = None
        linear_simi = x_train @ x_train.T
        params = " -t 4 -c {} -v 3 -q "
        for g in gs:
            rbf_simi = rbf_kernel(x_train, x_train, -g)
            custom_simi = linear_simi + rbf_simi
            custom_simi = np.hstack((np.arange(1, n_train + 1).reshape(-1, 1),
                                     custom_simi))
            prob = svm_problem(y_train, custom_simi, isKernel=True)
            for c in cs:
                param = svm_parameter(params.format(c))
                val_acc = svm_train(prob, param)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = params.format(c)
                    best_g = g

        print("best params: g = ", best_g, "  ", best_params)
        print("best val acc in training: ", best_val_acc)

        best_rbf_simi = rbf_kernel(x_train, x_train, -best_g)
        best_custom_simi = linear_simi + best_rbf_simi
        best_custom_simi = np.hstack((np.arange(1, n_train + 1).reshape(-1, 1),
                                      best_custom_simi))
        prob = svm_problem(y_train, best_custom_simi, isKernel=True)
        params = svm_parameter(best_params[:-8] + '-q ')
        model = svm_train(prob, params)

        n_test = x_test.shape[0]
        test_linear_simi = x_train @ x_test.T
        test_rbf_simi = rbf_kernel(x_train, x_test, -best_g)
        test_custom_simi = test_linear_simi + test_rbf_simi
        test_custom_simi = np.hstack((np.arange(1, n_test + 1).reshape(-1, 1),
                                      test_custom_simi.T))
        svm_predict(y_test, test_custom_simi, model)

    else:
        raise ValueError(f"Unknown part: {part}")


if __name__ == '__main__':
    main()
