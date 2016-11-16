import time, os, argparse
import numpy as np
from collections import Counter

from models import tf_glove
from util import clean_textfile, dump_corpus, get_regions_from_corpus

dir = os.path.dirname(os.path.realpath(__file__))

def main(args):
    textfile = args.textfile
    corpus = clean_textfile(textfile, args.stem)
    regions = get_regions_from_corpus(corpus)

    if args.debug is True:
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

    nb_search_iter = args.nb_search_iter
    num_epochs = args.num_epochs     

    log_dirs = []
    for i in range(nb_search_iter):
        # Random search
        if args.random_search is True:
            embedding_size = int(np.random.random_integers(200,500))
            context_size = int(np.random.random_integers(5,15))
        else:
            embedding_size = args.embedding_size
            context_size = args.context_size

        if args.debug is True:
            print('Init the GloVe model')
        glove = tf_glove.GloVeModel(embedding_size=embedding_size, context_size=context_size, learning_rate=1e-3)

        if args.debug is True:
            print('Fit to corpus and compute graph for training')
        glove.fit_to_corpus(regions)
        if args.debug is True:
            print('final vocab_size %d' % (glove.vocab_size))

        if args.debug is True:
            print('Start training for iter: %d' % i)
        log_dir = dir + '/results/glove/' + str(int(time.time()))
        log_dirs.append(log_dir)
        glove.train(num_epochs=num_epochs, log_dir=log_dir, summary_batch_interval=100, should_save=True)

    return log_dirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", nargs="?", const=True, default=False, type=bool, help="Debug mode (default: %(default)s)")
    parser.add_argument("--random_search", nargs="?", const=True, default=False, type=bool, help="Procede to a random search for hyperparameters tuning (default: %(default)s)")
    parser.add_argument("--textfile", default=dir + '/crawler/data/results.txt', type=str, help="The textfile to use build the corpus (default: %(default)s)")
    parser.add_argument("--embedding_size", default=200, type=int, help="Embedding size (default: %(default)s)")
    parser.add_argument("--context_size", default=10, type=int, help="Number of words to use for context (default: %(default)s)")
    parser.add_argument("--num_epochs", default=20, type=int, help="How many epochs should we train the GloVe (default: %(default)s)")
    parser.add_argument("--nb_search_iter", default=10, type=int, help="Batch size (default: %(default)s)")
    parser.add_argument("--stem", nargs="?", const=True, default=False, type=bool, help="Should we stem words? (default: %(default)s)")
    args = parser.parse_args()

    main(args)