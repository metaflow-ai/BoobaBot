import time, os, argparse
import numpy as np

import tf_glove
from util import clean_textfile, dump_corpus, get_corpus_with_paragraph

dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--debug", nargs="?", const=True, default=False, type=bool, help="debug mode (default: %(default)s)")

args = parser.parse_args()
if args.debug is True:
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

    print('Dumping cleaned results')
    cleaned_textfile = dir + '/crawler/data/results_clean.txt'
    dump_corpus(corpus, cleaned_textfile)

    nb_search_iter = 1
    embedding_size = 3
    context_size = 5
    num_epochs = 5
else:
    textfile = dir + '/crawler/data/results.txt'
    corpus = clean_textfile(textfile)
    corpus = get_corpus_with_paragraph(corpus)

    nb_search_iter = 10
    num_epochs = 80

for i in range(nb_search_iter):
    # Random search
    if args.debug is False:
        embedding_size = int(np.random.random_integers(100,300))
        context_size = int(np.random.random_integers(5,15))

    print('Init the GloVe model')
    model = tf_glove.GloVeModel(embedding_size=embedding_size, context_size=context_size, learning_rate=1e-3)

    print('Fit to corpus and compute graph for training')
    model.fit_to_corpus(corpus)
    print('final vocab_size %d' % (model.vocab_size))

    print('Start Training for iter: %d' % i)
    model.train(num_epochs=num_epochs, log_dir=dir + '/results/glove/' + str(int(time.time())), summary_batch_interval=100, should_save=True)

    #Todo Check for the best accuracy model and autonatically set it as a baseline