import time, os
import numpy as np

import tf_glove
from util import clean_textfile, dump_corpus, get_corpus_with_paragraph

dir = os.path.dirname(os.path.realpath(__file__))

textfile = dir + '/crawler/data/results.txt'
corpus = clean_textfile(textfile)
corpus = get_corpus_with_paragraph(corpus)
    
flatten_corpus = [y for x in corpus for y in x]
print("%d words found" % len(flatten_corpus))
nbTokens = len(set(flatten_corpus))
print('%d tokens found' % nbTokens)
chars = []
for word in flatten_corpus:
    chars += list(word)
chars = set(chars)
print(chars)

# print('Dumping ')
# cleaned_textfile = dir + '/crawler/data/results_clean.txt'
# dump_corpus(corpus, cleaned_textfile)
    
for i in range(10):
    # Random search
    embedding_size = int(np.random.random_integers(50,150))
    context_size = int(np.random.random_integers(5,10))

    print('Init the GloVe model')
    model = tf_glove.GloVeModel(embedding_size=embedding_size, context_size=context_size, learning_rate=1e-3)
    print('Fit to corpus and compute graph for training')
    model.fit_to_corpus(corpus)
    print('vocab_size %d' % (model.vocab_size))
    print('Start Training')
    model.train(num_epochs=500, log_dir=dir + '/results/' + str(int(time.time())), summary_batch_interval=100, should_save=True)
