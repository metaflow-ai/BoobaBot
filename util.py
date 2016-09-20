import re

# Takes a string and return a list of words
def clean_text(text):
    text = re.sub(r'(\w)\'(\w)', r'\1\' \2', text)
    text = re.sub(r'`|~|!|@|#|\$|%|\^|&|\*|\(|\)|_|\+|-|=|<|>|\?|/|\.|,|;|:|"|\\', '', text)
    cleaned_text = text \
        .lower() \
        .strip() \
        .split(' ')
    return cleaned_text

def clean_textfile(fullpath):
    with open(fullpath, 'r') as f:
        corpus = f.readlines()
        corpus = [clean_text(line) for line in corpus]

    return corpus

def get_corpus_with_paragraph(corpus):
    corpus_para = []
    region = []
    for line in corpus:
        if len(line) == 1 and line[0] == '':
            if len(region) != 0:
                corpus_para.append(region)
                region = []
            continue
        region += line

    return corpus_para

def dump_corpus(corpus, fullpath):
    with open(fullpath, 'w') as f:
        new_text = [' '.join(sublist) for sublist in corpus]
        new_text = '\n'.join(new_text)

        f.write(new_text)