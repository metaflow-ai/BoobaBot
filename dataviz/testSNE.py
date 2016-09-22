from sklearn import (manifold, decomposition, cluster)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import pprint

def plot_embedding(X):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        label = list(mapping.keys())[list(mapping.values()).index(i)]
        plt.text(1000*X[i, 0], 1000*X[i, 1], label, size=3, zorder=1, color='k')


    plt.axis([0, 1000, 0, 1000])
    plt.savefig('foo.eps', format='eps', dpi=1000)


def plot_embedding_k_mean(X, k_means_labels, k_means_cluster_centers):

    # Center and give some space
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = 10000 * (X - x_min) / (x_max - x_min)
    k_means_cluster_centers = 10000 * (k_means_cluster_centers - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)

    finalClusters = len(k_means_cluster_centers)

    # Generate 10 colors
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, finalClusters))

    for k, col in zip(range(finalClusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.', markersize=3)
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=5)


    for i in range(X.shape[0]):
        label = list(mapping.keys())[list(mapping.values()).index(i)]
        plt.text(X[i, 0], X[i, 1], label, size=3, zorder=2, color='k')


    plt.axis([0, 10000, 0, 10000])
    plt.axis('off')
    plt.savefig('foo.eps', format='eps', dpi=3000, bbox_inches='tight')

with open('data.json') as jsonData:
    rawData = json.load(jsonData)

    size = rawData['vocab_size']
    print('This dataset has a size of %d data points' % (size))

    mapping = rawData['word_to_id']
    X = rawData['embed']
    X = X[1000:2000]

    print('Computing PCA')
    pca = decomposition.PCA(n_components=50)
    X_pca = pca.fit_transform(X)
    print('Computed PCA')

    print('Computing tSNE')
    tsne = manifold.TSNE(n_components=2, init="pca", random_state=0)
    X_tsne = tsne.fit_transform(X_pca)
    print('Computed tSNE')

    print('K-MEAN')
    kMeans = cluster.KMeans(init='k-means++', n_clusters=25, n_init=10)
    kMeans.fit(X_tsne)

    k_means_labels = kMeans.labels_
    k_means_cluster_centers = kMeans.cluster_centers_
    # k_means_labels_unique = np.unique(k_means_labels)

    print('Plot')
    # plot_embedding(X_tsne)
    plot_embedding_k_mean(X_tsne, k_means_labels, k_means_cluster_centers)



