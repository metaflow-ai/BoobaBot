import time, os, re
import numpy as np

import tf_glove

dir = os.path.dirname(os.path.realpath(__file__))

# Takes a string and return a list of words
def clean_text(text):
    text = re.sub(r'(\w)\'(\w)', r'\1\' \2', text)
    text = re.sub(r'`|~|!|@|#|\$|%|\^|&|\*|\(|\)|_|\+|-|=|<|>|\?|/|\.|,|;|:|"|\\', '', text)
    cleaned_text = text \
        .lower() \
        .strip() \
        .split(' ')
    return cleaned_text

with open(dir + '/crawler/data/results.txt', 'r') as f:
    corpus = f.readlines()
    corpus = [clean_text(line) for line in corpus]

    print(len(set([y for x in corpus for y in x])))

    with open(dir + '/crawler/data/results_clean.txt', 'w') as f:
        new_text = [' '.join(sublist) for sublist in corpus]
        new_text = '\n'.join(new_text)

        f.write(new_text)


for i in range(10):
    # Random search
    embedding_size = int(np.random.random_integers(50,150))
    context_size = int(np.random.random_integers(1,10))

    print('Init the GloVe model')
    model = tf_glove.GloVeModel(embedding_size=embedding_size, context_size=context_size, learning_rate=1e-3)
    print('Fit to corpus and compute graph for training')
    model.fit_to_corpus(corpus)
    print('vocab_size %d' % (model.vocab_size))
    print('Start Training')
    model.train(num_epochs=60, log_dir=dir + '/results/' + str(int(time.time())), summary_batch_interval=100, should_save=True)
