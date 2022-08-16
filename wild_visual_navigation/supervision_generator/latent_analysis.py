from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.learning.dataset import LocomotionLatentDataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import umap

if __name__ == "__main__":
    root = str(os.path.join(WVN_ROOT_DIR, "assets/locomotion_latent"))
    latent_filename = "latent.csv"
    dataset = LocomotionLatentDataset(root, latent_filename)
    df = dataset.get_dataframe()

    X = df.loc[:, ((df.columns != "ts") & (df.columns != "sec") & (df.columns != "nsec"))]
    X = StandardScaler().fit_transform(X.values)

    # # Plot all components
    # plt.plot(X)
    # plt.legend()
    # plt.show()

    # PCA
    N = 3
    palette = sns.color_palette("colorblind", N)

    pca = PCA(n_components=N)
    pca.fit(X)
    X_pca = pca.transform(X)
    fig, axs = plt.subplots(N, 1, sharex=True)
    for i in range(N):
        axs[i].plot(df["ts"], X_pca[:, i], alpha=1, label=f"comp_{i}", color=palette[i])
        axs[i].legend(loc="upper right")
        axs[1].set_ylabel("Component")

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

    # # Kernel PCA
    # kpca = KernelPCA(n_components=3)
    # kpca.fit(X)
    # X_kpca = kpca.transform(X)

    # plt.plot(df["ts"], X_kpca, alpha=0.5)
    # plt.show()

    # t-SNE
    # X_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
    # plt.plot(df["ts"], X_tsne, alpha=0.5)
    # plt.show()

    # # UMAP
    # reducer = umap.UMAP()
    # x_umap = reducer.fit_transform(X_scaled)
    # plt.plot(df["ts"], x_umap, alpha=0.5)
    # plt.show()

    # print(X_embedded)
