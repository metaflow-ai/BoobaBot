import re, os, json
import numpy as np

from nltk.stem.snowball import FrenchStemmer, SnowballStemmer
from nltk.tokenize import WordPunctTokenizer


# sentenceTokenizer = nltk.data.load('tokenizers/punkt/PY3/french.pickle')
stemmer = SnowballStemmer("french")
tokenizer = WordPunctTokenizer()
# stemmer = FrenchStemmer()

# Takes a string and return a list of words
def clean_text(text):
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
    # Characters whitelist to avoid any unknkown chars
    text = re.sub(r'[^a-zA-Z0-9 àáâãäçèéêëìíîïñòóôõöùúûüýÿ\'"\.,?;:\'"!-]', '', text) 
    

    tokens = tokenizer.tokenize(text)
    cleaned_tokens = [stemmer.stem(w) for w in tokens]

    return cleaned_tokens

def clean_textfile(fullpath):
    with open(fullpath, 'r') as f:
        corpus = f.readlines()
        corpus = [clean_text(line) for line in corpus]

    return corpus

def get_corpus_with_paragraph(corpus):
    corpus_para = []
    region = []
    for line in corpus:
        if len(line) == 0:
            if len(region) != 0:
                corpus_para.append(region)
                region = []
            continue
        region += line

    if len(region) != 0:
        corpus_para.append(region)

    return corpus_para

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


def make_sets(corpus, wti_dict):
    np.random.shuffle(corpus)
    nb_para = len(corpus)
    nb_para_dev_test = nb_para // 10
    nb_para_train = nb_para - 2 * nb_para_dev_test

    train_data = corpus[:nb_para_train]
    train_data = [word_to_id(wti_dict, y) for x in train_data for y in x]

    dev_data = corpus[nb_para_train:nb_para_train + nb_para_dev_test]
    dev_data = [word_to_id(wti_dict, y) for x in dev_data for y in x]

    test_data = corpus[nb_para_train + nb_para_dev_test:]
    test_data = [word_to_id(wti_dict, y) for x in test_data for y in x]

    return train_data, dev_data, test_data

def load_corpus_as_sets(fullpath, wti_dict):
    corpus = clean_textfile(fullpath)
    corpus = get_corpus_with_paragraph(corpus)
    return make_sets(corpus, wti_dict)


def evaluate_recall(y_pred, labels, k=1):
    num_examples = float(len(y_pred))
    num_correct = 0
    for preds, label in zip(y_pred, labels):
        if label in preds[:k]:
            num_correct += 1
    return num_correct / num_examples

def predict_random(utterances):
    return np.random.choice(len(utterances),10, replace=False)

def print_learningconfig():
    for subdir, dirs, files in os.walk('results'):
        for file in files:
            if file == 'data.json':
                path = os.path.join(subdir, file)
                print(path)
                with open(path) as jsonData:
                    data = json.load(jsonData)
                    print(data['learning_config'])