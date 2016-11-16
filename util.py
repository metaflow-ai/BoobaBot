import re, os, json, operator, functools
import numpy as np
from collections import Counter, deque

from nltk.stem.snowball import FrenchStemmer, SnowballStemmer
from nltk.tokenize import WordPunctTokenizer


# sentenceTokenizer = nltk.data.load('tokenizers/punkt/PY3/french.pickle')
stemmer = SnowballStemmer("french")
tokenizer = WordPunctTokenizer()
# stemmer = FrenchStemmer()

# Takes a string and return a list of words
def clean_text(text, stem=False):
    # Removing curly braces, those are metadata in the corpus
    text = re.sub(r'\{.*}', '', text)
    # Remove x2, x3 etc. (repeating verse annotation)
    text = re.sub(r'(x|X)\d+', '', text)
    # Replacing purely stylistics chars
    text = re.sub(r'æ', 'ae', text)
    text = re.sub(r'œ', 'oe', text)
    text = re.sub(r'[ìíîï]', 'i', text)
    text = re.sub(r'[ýÿ]', 'y', text)
    text = re.sub(r'[òóôõö]', 'o', text)
    text = re.sub(r'[áâãä]', 'a', text)
    text = re.sub(r'ë', 'e', text)
    text = re.sub(r'ñ', 'n', text)
    text = re.sub(r'[ûü]', 'u', text)
    text = re.sub(r'[«“”»]', '"', text)
    text = re.sub(r'[…]', '...', text)
    # Characters whitelist to avoid any unknkown chars
    text = re.sub(r'[^a-zA-Z0-9 àáâãäçèéêëìíîïñòóôõöùúûüýÿ\'"\.,?;:\'"!-]', '', text)


    tokens = tokenizer.tokenize(text)
    if stem is True:
        cleaned_tokens = [stemmer.stem(w) for w in tokens]
    else:
        cleaned_tokens = [w.lower() for w in tokens]

    return cleaned_tokens

def clean_textfile(fullpath, stem=False):
    EOP = False
    with open(fullpath, 'r') as f:
        corpus = f.readlines()
        cleaned_corpus = []
        for line in corpus:
            cleaned_line = clean_text(line, stem)
            if len(cleaned_line) != 0:
                cleaned_line += ['<EOL>']
                EOP = False
                cleaned_corpus += cleaned_line
            else:
                # If EOP is True, it means we have multiple empty line
                if EOP is False:
                    cleaned_line = ['<EOP>']
                    EOP = True
                    cleaned_corpus += cleaned_line
                
    if EOP is False:
        cleaned_corpus += ['<EOP>']

    return cleaned_corpus

def get_regions_from_corpus(corpus):
    regions = []
    region = []
    for word in corpus:
        region.append(word)
        if word == '<EOP>':
            regions.append(region)
            region = []

    return regions

def dump_corpus(corpus, fullpath):
    with open(fullpath, 'w') as f:
        new_text = [' '.join(sublist) for sublist in corpus]
        new_text = '\n'.join(new_text)

        f.write(new_text)

def word_to_id(wti_dict, word):
    if word in wti_dict:
        return wti_dict[word]
    else:
        return wti_dict['<UNK>']


def make_sets(corpus, wti_dict, dev_test_size=0.1):
    counter = Counter()
    counter.update(corpus)
    
    nb_para = counter['<EOP>']
    nb_para_dev_test = int(nb_para * dev_test_size)
    nb_para_train = nb_para - 2 * nb_para_dev_test

    para_indexes = [i for i,word in enumerate(corpus) if word == '<EOP>']
    end_train_set_index = para_indexes[nb_para_train - 1] 
    end_dev_set_index = para_indexes[nb_para_train + nb_para_dev_test - 1] 

    train_set = corpus[:end_train_set_index]
    train_set = [word_to_id(wti_dict, word) for word in train_set]

    dev_set = corpus[end_train_set_index:end_dev_set_index]
    dev_set = [word_to_id(wti_dict, word) for word in dev_set]

    test_set = corpus[end_dev_set_index:]
    test_set = [word_to_id(wti_dict, word) for word in test_set]

    return train_set, dev_set, test_set

def load_corpus_as_sets(fullpath, wti_dict):
    corpus = clean_textfile(fullpath)
    return make_sets(corpus, wti_dict)

def print_learningconfig():
    for subdir, dirs, files in os.walk('results'):
        for file in files:
            if file == 'config.json':
                path = os.path.join(subdir, file)
                print(path)
                with open(path) as jsonData:
                    data = json.load(jsonData)
                    print(data['config'])

def get_nb_parameters(tf_var_list):
    nb_params = 0
    for t_var in tf_var_list:
        shape = t_var.get_shape().as_list()
        if type(shape[0]) != int:
            shape.pop(0)    
        nb_params += get_nb_elements_from_shape(shape)

    return nb_params

def get_nb_elements_from_shape(shape):
    if len(shape) == 0:
        return 0

    return functools.reduce(operator.mul, shape)
