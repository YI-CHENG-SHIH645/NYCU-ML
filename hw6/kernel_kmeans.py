from utils import Clustering


if __name__ == '__main__':
    # 2 images X 3 types of clustering
    Clustering(tgt_img="image1.png",
               method="Kernel_KMeans",
               n_clusters=2,
               init_method="random").clustering()
