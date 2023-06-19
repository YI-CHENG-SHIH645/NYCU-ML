import os
import imageio
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform

colors = ['black', 'darkorange', 'cornflowerblue',
          'silver', 'gold', 'green', 'navy',
          'magenta', 'yellow', 'red']


class Clustering:
    def __init__(self,
                 tgt_img: str,
                 method: str,
                 n_clusters: int,
                 init_method: str,
                 cut: str = "",
                 gamma_s=1/(100*100),
                 gamma_c=1/(255*255)):
        assert method in ["Kernel_KMeans", "Spectral"]
        assert init_method in ["random", "kmeans++"]
        assert cut in ["", "ratio", "normalized"]
        self.method = method
        self.init_method = init_method
        self.img = iio.imread(tgt_img)
        self.tgt_img = tgt_img.split(".")[0]
        self.img_coord = np.array(
            np.meshgrid(np.arange(self.img.shape[0]), np.arange(self.img.shape[1]))
        ).T.reshape(-1, 2)
        self.n_clusters = n_clusters
        self.gamma_s = gamma_s
        self.gamma_c = gamma_c
        self.num_pixels = self.img.shape[0] * self.img.shape[1]

        # for Spectral
        self.cut = cut

        self.K = None

        # which cluster for each pixel
        self.grps = None

        # path to store experiment results
        self.file_dir = None

        self.init(init_method)

    def get_kernel(self):
        img_flattened = self.img.reshape(self.num_pixels, 3)

        # spatial similarity
        K_s = squareform(np.exp(-self.gamma_s * pdist(self.img_coord, metric='sqeuclidean')))
        np.fill_diagonal(K_s, 1.0)

        # color similarity
        K_c = squareform(np.exp(-self.gamma_c * pdist(img_flattened, metric='sqeuclidean')))
        np.fill_diagonal(K_c, 1.0)

        # Gram matrix
        K = K_s * K_c

        return K

    def init(self, init_method: str):
        self.K = self.get_kernel()

        # randomly assign a cluster to each pixel
        self.grps = np.random.randint(self.n_clusters, size=self.num_pixels)

        self.file_dir = f"./{self.method}/{self.tgt_img}_{self.n_clusters}c_{init_method}_{self.cut}"
        os.makedirs(self.file_dir, exist_ok=True)
        for file in os.listdir(self.file_dir):
            os.remove(os.path.join(self.file_dir, file))

    def clustering(self):
        if self.method == "Kernel_KMeans":
            # record distance of each pixel to cluster centroid
            dist = np.zeros((self.num_pixels, self.n_clusters))
            for it in range(1000):
                dist.fill(0)
                for grp in range(self.n_clusters):
                    m = self.grps == grp
                    # mean of within group data points as cluster centroid
                    dist[:, grp] += np.mean(self.K[m][:, m])
                    # calculate distance of all pixels -> this cluster centroid
                    dist[:, grp] -= 2 * np.mean(self.K[:, m], axis=1)
                old_grps = self.grps
                # based on new evaluated distances, reassign cluster for each pixel
                self.grps = new_grps = np.argmin(dist, axis=1)

                self.plt_clustered_img(it)

                num_changed = np.count_nonzero(new_grps - old_grps)
                if num_changed / self.num_pixels < 1e-3:
                    print(f"Converge at iter: {it}")
                    break
            self.draw_gif()

        elif self.method == "Spectral":
            D = self.K.sum(axis=1)
            if self.cut == "ratio":
                L = np.diag(D) - self.K
            elif self.cut == "normalized":
                D = np.diag(D**-0.5)
                L = D @ self.K @ D
            else:
                raise ValueError(f"Unknown cut : {self.cut}")

            i_val, i_vec = np.linalg.eig(L)
            # nth column is the nth (largest eigen value) eigen vector
            U = i_vec[:, np.argsort(i_val)[::-1]]

            # pick top `n_clusters` eigen vector
            # due to floating point precision, i_val may contain complex value
            # use .real to treat it as 0
            U = U[:, :self.n_clusters].real

            # normalize each row vector as unit length vector
            U /= np.sqrt((U ** 2).sum(axis=1, keepdims=True))

            if self.init_method == "random":
                # pick `n_clusters` data points as centroids randomly from all pixels
                centroids = U[np.random.choice(self.num_pixels, size=self.n_clusters, replace=False), :]
            elif self.init_method == 'kmeans++':
                centroids = [U[np.random.choice(self.num_pixels), :]]
                # carefully choose the initial centroid,
                # The goal is to make the centroids spread out as much as possible
                for i in range(self.n_clusters - 1):
                    dist = np.min(cdist(U, centroids, 'euclidean'), axis=1)
                    prob = dist / np.sum(dist)
                    # pick one centroid, the more far, the more likely to be picked
                    centroids.append(U[np.random.choice(self.num_pixels, p=prob)])
                centroids = np.array(centroids)
            else:
                raise ValueError(f"Unknown init method : {self.init_method}")

            for it in range(1000):
                # distance of all points to all centroids
                p2c = cdist(U, centroids, metric='euclidean')
                old_grps = self.grps
                # reassign groups(clusters) to all points
                self.grps = new_grps = np.argmin(p2c, axis=1)
                num_changed = np.count_nonzero(new_grps - old_grps)

                # update centroid
                for i in range(self.n_clusters):
                    m = self.grps == i
                    centroids[i, :] = np.mean(U[m, :], axis=0)

                self.plt_clustered_img(it)

                if num_changed / self.num_pixels < 1e-3:
                    break

            self.draw_gif()
            if self.n_clusters == 2:
                self.plt_eigen_space(U)
        else:
            raise ValueError(f"Unknown method : {self.method}")

    def plt_clustered_img(self, nth_iter: int):
        plt.subplot(121, aspect='equal')
        plt.imshow(self.img)
        n = int(np.sqrt(self.num_pixels))
        plt.subplot(122, aspect='equal')

        for grp in range(self.n_clusters):
            m = self.grps == grp
            plt.scatter(self.img_coord[m][:, 1], self.img_coord[m][:, 0], s=2, color=colors[grp])
        plt.xlim(0, n)
        plt.ylim(n, 0)

        file_name = f"iter_{nth_iter}"
        plt.suptitle(file_name)
        plt.tight_layout()
        plt.savefig(os.path.join(self.file_dir, file_name + ".png"))

    def draw_gif(self):
        gif_file_path = os.path.join(self.file_dir, "result.gif")
        with imageio.get_writer(gif_file_path, mode='I', duration=0.5) as writer:
            for filename in sorted(os.listdir(self.file_dir)):
                image = iio.imread(os.path.join(self.file_dir, filename))
                writer.append_data(image)

    def plt_eigen_space(self, U):
        assert self.method == "Spectral" and self.n_clusters == 2
        for grp in range(self.n_clusters):
            m = self.grps == grp
            plt.scatter(U[m, 0], U[m, 1], s=2, color=colors[grp])
        plt.title("Eigen Space")
        plt.tight_layout()
        plt.savefig(os.path.join(self.file_dir, "eigen_space.png"))
