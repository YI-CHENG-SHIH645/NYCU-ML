import re
import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist


def read_faces(resize_shape=(50, 50)):
    training_faces = []
    training_lbl = []
    testing_faces = []
    testing_lbl = []

    with zipfile.ZipFile("ML_HW07/Yale_Face_Database.zip") as face_zip:
        for filename in face_zip.namelist():
            if not filename.endswith(".pgm"):
                continue  # not a face picture
            with face_zip.open(filename) as image:
                # If we extracted files from zip, we can use cv2.imread(filename) instead
                if filename.split("/")[1] == "Training":
                    training_faces.append(
                        cv2.resize(
                            cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE),
                            resize_shape,
                            interpolation=cv2.INTER_AREA
                        ).ravel()
                    )
                    training_lbl.append(int(re.findall(r'subject(\d+)', filename)[0]))
                else:
                    testing_faces.append(
                        cv2.resize(
                            cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE),
                            resize_shape,
                            interpolation=cv2.INTER_AREA
                        ).ravel()
                    )
                    testing_lbl.append(int(re.findall(r'subject(\d+)', filename)[0]))

    return np.vstack(training_faces), np.array(training_lbl),\
        np.vstack(testing_faces), np.array(testing_lbl)


def eig_decom(M):
    eig_vals, eig_vecs = np.linalg.eig(M)
    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real
    desc_eig_vals_arg = np.argsort(-eig_vals)
    return eig_vals[desc_eig_vals_arg], eig_vecs[:, desc_eig_vals_arg]


class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.X_mean = 0
        self.eig_vecs = None

    def fit(self, X, y=None):
        self.X_mean = np.mean(X, axis=0, keepdims=True)
        X = X - self.X_mean
        cov = X @ X.T
        # cov = X.T @ X
        eig_vals, eig_vecs = eig_decom(cov)
        # 135 dimension -> 25 dimension
        eig_vecs = (X - self.X_mean).T @ eig_vecs[:, :self.n_components]
        # we want a linear combination of eigen faces to reconstruct original faces
        # eig_faces(eig vecs) have to be unit length so that the
        # projection length = inner product
        self.eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=0, keepdims=True)

        return self

    def transform(self, X):
        return (X-self.X_mean) @ self.eig_vecs

    def reconstruct(self, z):
        return z @ self.eig_vecs.T + self.X_mean


class KernelPCA(PCA):
    def __init__(self, n_components: int, kernel: str):
        super().__init__(n_components)
        self.kernel = kernel
        self.K = None
        self.X = None
        self.rbf = lambda x, x_prime: np.exp(-1e-7*dist.cdist(x, x_prime, metric='sqeuclidean'))
        self.poly = lambda x, x_prime: np.power(2*(x @ x_prime.T)+5, 3)

    def fit(self, X, y=None):
        self.X = X
        K = getattr(self, self.kernel)(X, X)
        sq1 = np.ones_like(K) / K.shape[0]
        self.K = K - sq1@K - K@sq1 + sq1@K@sq1
        eig_vals, eig_vecs = eig_decom(K)
        eig_vecs = eig_vecs[:, :self.n_components]
        self.eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=0, keepdims=True)

        return self

    def transform(self, X):
        return getattr(self, self.kernel)(X, self.X) @ self.eig_vecs

    def reconstruct(self, z):
        raise NotImplementedError


class LDA(PCA):
    def __init__(self, n_components: int):
        super().__init__(n_components)

    def fit(self, X, y=None):
        assert y is not None
        D = X.shape[1]
        n_groups = np.unique(y).size
        grp_means = np.zeros((n_groups, D))
        grp_sizes = np.zeros((n_groups, 1), dtype=int)
        S_within = np.zeros(shape=(D, D))
        for grp in np.unique(y):
            grp_idx = y == grp
            grp_sizes[grp-1] = np.count_nonzero(grp_idx)
            grp_means[grp-1] = np.mean(X[grp_idx], axis=0)
            S_within += (X[grp_idx] - grp_means[grp-1]).T @ (X[grp_idx] - grp_means[grp-1])
        mean = np.mean(X, axis=0, keepdims=True)
        S_between = (grp_means-mean).T @ (grp_sizes*(grp_means-mean))
        eig_vals, eig_vecs = eig_decom(np.linalg.inv(
            S_within+1e-8*np.eye(S_within.shape[0])) @ S_between)
        eig_vecs = eig_vecs[:, :self.n_components].real
        self.eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=0, keepdims=True)

        return self


class KernelLDA(KernelPCA):
    def __init__(self, n_components: int, kernel: str):
        super().__init__(n_components, kernel)

    def fit(self, X, y=None):
        self.X = X
        N = X.shape[0]
        n_groups = np.unique(y).size
        grp_means = np.zeros((n_groups, N))
        grp_sizes = np.zeros((n_groups, 1), dtype=int)
        S_within = np.zeros((N, N))
        for grp in np.unique(y):
            grp_idx = y == grp
            grp_sizes[grp - 1] = np.count_nonzero(grp_idx)
            K_tot2grp = getattr(self, self.kernel)(X, X[grp_idx])
            grp_means[grp-1] = np.mean(K_tot2grp, axis=1)
            n_in_cls = grp_sizes[grp - 1, 0]
            H_c = np.eye(n_in_cls) - (1 / n_in_cls) * np.ones((n_in_cls, n_in_cls))
            S_within += K_tot2grp @ H_c @ K_tot2grp.T
        K = getattr(self, self.kernel)(X, X)
        mean_K = np.mean(K, axis=0, keepdims=True)
        S_between = (grp_means - mean_K).T @ (grp_sizes * (grp_means - mean_K))
        eig_vals, eig_vecs = eig_decom(np.linalg.inv(
            S_within + 1e-8 * np.eye(S_within.shape[0])) @ S_between)
        eig_vecs = eig_vecs[:, :self.n_components].real
        self.eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=0, keepdims=True)

        return self

    def reconstruct(self, z):
        raise NotImplementedError


def face_recognition(z, z_lbl, z_test, z_test_lbl, method: str, k: int = 5):
    dist_matrix = dist.cdist(z_test, z, metric='sqeuclidean')
    arg_sorted = np.argsort(dist_matrix, axis=1)
    knn_lbl = z_lbl[arg_sorted][:, :k]
    pred_lbl = np.array([np.argmax(np.bincount(lbl)) for lbl in knn_lbl])
    num_correct = np.count_nonzero(z_test_lbl == pred_lbl)
    print(f"{method} Accuracy : {num_correct/z_test.shape[0]:.3f}")


def plot_image_grid(images, resize_shape: tuple, title=""):
    N = images.shape[0]
    n_rows, n_cols = int(np.ceil(N / 5)), 5
    fig = plt.figure(figsize=(8, 8 * n_rows / n_cols))
    axes = [fig.add_subplot(n_rows, n_cols, i + 1) for i in range(N)]
    for i, ax in enumerate(axes):
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.patch.set_edgecolor("black")
        ax.patch.set_linewidth(2)
        ax.imshow(images[i].reshape(resize_shape), cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(title, fontsize=16)


def main():
    resize_shape = (50, 50)
    training_faces, training_lbl, testing_faces, testing_lbl = read_faces(resize_shape=resize_shape)
    idxes = np.random.choice(np.arange(len(training_faces)), 10)

    plot_image_grid(training_faces[idxes], resize_shape=resize_shape, title="original faces")

    pca = PCA(n_components=25).fit(training_faces)
    z = pca.transform(training_faces)
    training_faces_prime = pca.reconstruct(z)
    face_recognition(z, training_lbl, pca.transform(testing_faces), testing_lbl,
                     method="PCA", k=5)

    plot_image_grid(pca.eig_vecs.T, resize_shape=resize_shape, title="pca eigen faces")
    plot_image_grid(training_faces_prime[idxes], resize_shape=resize_shape, title="pca recovered faces")

    lda = LDA(n_components=25).fit(training_faces, training_lbl)
    z = lda.transform(training_faces)
    training_faces_prime = lda.reconstruct(z)
    face_recognition(z, training_lbl, lda.transform(testing_faces), testing_lbl,
                     method="LDA", k=5)

    plot_image_grid(lda.eig_vecs.T, resize_shape=resize_shape, title="lda eigen faces")
    plot_image_grid(training_faces_prime[idxes], resize_shape=resize_shape, title="lda recovered faces")

    k_pca = KernelPCA(n_components=25, kernel="rbf").fit(training_faces)
    z = k_pca.transform(training_faces)
    face_recognition(z, training_lbl, k_pca.transform(testing_faces), testing_lbl,
                     method="kernel(rbf) PCA", k=5)

    k_lda = KernelLDA(n_components=25, kernel="rbf").fit(training_faces, training_lbl)
    z = k_lda.transform(training_faces)
    face_recognition(z, training_lbl, k_lda.transform(testing_faces), testing_lbl,
                     method="kernel(rbf) LDA", k=5)

    k_pca = KernelPCA(n_components=25, kernel="poly").fit(training_faces)
    z = k_pca.transform(training_faces)
    face_recognition(z, training_lbl, k_pca.transform(testing_faces), testing_lbl,
                     method="kernel(polynomial) PCA", k=5)

    k_lda = KernelLDA(n_components=25, kernel="poly").fit(training_faces, training_lbl)
    z = k_lda.transform(training_faces)
    face_recognition(z, training_lbl, k_lda.transform(testing_faces), testing_lbl,
                     method="kernel(polynomial) LDA", k=5)

    # plt.show()


if __name__ == '__main__':
    main()
