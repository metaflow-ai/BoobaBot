from sklearn import (manifold, decomposition, cluster)
import numpy as np
import tensorflow as tf
import json, matplotlib, os, argparse, collections
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dir = os.path.dirname(os.path.realpath(__file__))

def load_embedding(config):
    embedding = tf.get_variable(
        'embedding',
        shape=[config['vocab_size'], config['embedding_size']],
        initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0)
    )

    embedding_saver = tf.train.Saver({
        config['embedding_var_name']: embedding
    })
    embedding_chkpt = tf.train.get_checkpoint_state(config['glove_dir'])
    with tf.Session() as sess:
        embedding_saver.restore(sess, save_path=embedding_chkpt.model_checkpoint_path)
        embedding_value = sess.run(embedding)

    return embedding_value

def plot_embedding(X, plot_name, mapping):
    results_dir = dir + '/results'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    print('Plotting without k-means')
    plt.figure()
    ax = plt.subplot(111)

    for i, label in enumerate(mapping):
        # print(X[i, 0], X[i, 1], label)
        plt.text(X[i, 0], X[i, 1], label, size=3, zorder=1, color='k')

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    plt.axis([mins[0], maxs[0], mins[1], maxs[1]])
    plt.show()
    
    plt.savefig(results_dir + '/' + plot_name + '.eps', format='eps', dpi=1000)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_dir", default="results/glove", type=str, help="glove dir (default: %(default)s)")
    parser.add_argument("--troisd", nargs="?", const=True, default=False, type=bool, help="3d mode for t-SNE (default: %(default)s)")
    args = parser.parse_args()

    glove_dir = dir + '/../' + args.glove_dir
    glove_name = args.glove_dir.split('/')[-1]
    if glove_name == '':
        glove_name = args.glove_dir.split('/')[-2]

    with open(glove_dir + '/config.json') as jsonData:
        rawData = json.load(jsonData)
        config = dict(rawData['config'])
        config['glove_dir'] = glove_dir
        
    print('Ordering mapping')
    mapping = rawData['word_to_id_dict']
    mapping = {value:keys for keys, value in zip(list(mapping.keys()), list(mapping.values())) }
    mapping = collections.OrderedDict(sorted(mapping.items()))
    mapping = list(mapping.values())

    print('Loading embedding of dimensions vocab:%d x embed_size:%d' % (config['vocab_size'], config['embedding_size']))
    X = load_embedding(config)
    if len(X) > 2000:
        print('Drawing 1000 random samplew between 1000 and 2000')
        X = X[1000:2000]
        mapping = mapping[1000:2000]

    print('Computing full PCA for variance > 90%')
    pca = decomposition.PCA(
        n_components=0.9,
        svd_solver='full'
        )
    X_pca = pca.fit_transform(X)
    print('Computed PCA with shape: ', X_pca.shape)

    if args.troisd is True:
        print('Computing t-SNE in 3D')
        tsne_components = 3
    else:
        print('Computing t-SNE in 2D')
        tsne_components = 2
    X_tsnes = []
    for p in [2, 5, 10 ,30, 50, 100]:
        print('Computing t-SNE with perplexity %d and init PCA' % p)
        tsne = manifold.TSNE(
            n_components=tsne_components, 
            perplexity=p, 
            init="pca",
            n_iter=5000, 
            )
        X_tsnes.append([tsne.fit_transform(X_pca), '%s-tsne-p%d' % (glove_name, p)])
        print('Computed tSNE')

    # print('K-MEAN')
    # kMeans = cluster.KMeans(init='k-means++', n_clusters=25, n_init=10)
    # kMeans.fit(X_tsne)

    # k_means_labels = kMeans.labels_
    # k_means_cluster_centers = kMeans.cluster_centers_
    # # k_means_labels_unique = np.unique(k_means_labels)

    for X_tsne in X_tsnes:
        plot_embedding(X_tsne[0], X_tsne[1], mapping)
    # plot_embedding_k_mean(X_tsne, k_means_labels, k_means_cluster_centers)



