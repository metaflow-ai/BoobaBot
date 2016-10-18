import time, os, argparse
import numpy as np
from collections import Counter

import tf_glove
from util import clean_textfile, dump_corpus, get_regions_from_corpus

dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--debug", nargs="?", const=True, default=False, type=bool, help="debug mode (default: %(default)s)")
parser.add_argument("--num_epochs", default=20, type=int, help="How many epochs should we train the GloVe (default: %(default)s)")
parser.add_argument("--nb_search_iter", default=10, type=int, help="Batch size (default: %(default)s)")
parser.add_argument("--stem", nargs="?", const=True, default=False, type=bool, help="debug mode (default: %(default)s)")


args = parser.parse_args()
if args.debug is True:
    textfile = dir + '/crawler/data/test_results.txt'
    corpus = clean_textfile(textfile, args.stem)
    print("%d words found" % len(corpus))

    counter = Counter()
    counter.update(corpus)
    print('%d paragraphe found' % counter['<EOP>'])
    
    nb_tokens = len(set(corpus))
    print('%d unique tokens found' % nb_tokens)

    chars = set(''.join(corpus))
    print('%d chars found' % len(chars))
    print(chars)

    print('Dumping cleaned results')
    cleaned_textfile = dir + '/crawler/data/results_clean.txt'
    dump_corpus(corpus, cleaned_textfile)

    regions = get_regions_from_corpus(corpus)

    nb_search_iter = 1
    embedding_size = 3
    context_size = 4
    num_epochs = 20
else:
    textfile = dir + '/crawler/data/results.txt'
    corpus = clean_textfile(textfile, args.stem)
    regions = get_regions_from_corpus(corpus)

    nb_search_iter = args.nb_search_iter
    num_epochs = args.num_epochs

for i in range(nb_search_iter):
    # Random search
    if args.debug is False:
        embedding_size = int(np.random.random_integers(200,500))
        context_size = int(np.random.random_integers(5,15))

    print('Init the GloVe model')
    glove = tf_glove.GloVeModel(embedding_size=embedding_size, context_size=context_size, learning_rate=1e-3)

    print('Fit to corpus and compute graph for training')
    glove.fit_to_corpus(regions)
    print('final vocab_size %d' % (glove.vocab_size))

    print('Start training for iter: %d' % i)
    glove.train(num_epochs=num_epochs, log_dir=dir + '/results/glove/' + str(int(time.time())), summary_batch_interval=100, should_save=True)