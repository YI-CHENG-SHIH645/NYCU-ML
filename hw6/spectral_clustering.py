from utils import Clustering


if __name__ == '__main__':
    # 2 images X 3 types of clustering X 2 types of cut
    Clustering(tgt_img="image1.png",
               method="Spectral",
               n_clusters=2,
               init_method="random",
               cut="normalized").clustering()
