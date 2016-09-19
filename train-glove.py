import time, os
import numpy as np

import tf_glove

dir = os.path.dirname(os.path.realpath(__file__))
with open(dir + '/crawler/data/results.txt', 'r') as f:
    corpus = f.readlines()
    for i,line in enumerate(corpus):
        corpus[i] = line.lower().strip().split(' ')

for i in range(10):
    # Random search
    embedding_size = int(np.random.random_integers(50,300))
    context_size = int(np.random.random_integers(3,10))

    print('Init the GloVe model')
    model = tf_glove.GloVeModel(embedding_size=embedding_size, context_size=context_size, learning_rate=1e-3)
    print('Fit to corpus and compute graph for training')
    model.fit_to_corpus(corpus)
    print('Start Training')
    model.train(num_epochs=600, log_dir=dir + '/results/' + str(int(time.time())), summary_batch_interval=100, should_save=True)
    model.print_sum_up()
