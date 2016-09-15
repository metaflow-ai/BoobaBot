import time, os
import tf_glove

dir = os.path.dirname(os.path.realpath(__file__))
with open(dir + '/crawler/data/results.txt', 'r') as f:
    corpus = f.readlines()
    for i,line in enumerate(corpus):
        corpus[i] = line.lower().strip().split(' ')

print('Init the GloVe model')
model = tf_glove.GloVeModel(embedding_size=200, context_size=10, learning_rate=1e-3)
print('Fit to corpus and compute graph for training')
model.fit_to_corpus(corpus)
print('Start Training')
model.train(num_epochs=500, log_dir=dir + '/results/' + str(int(time.time())), summary_batch_interval=100, should_save=True)
model.print_sum_up()
