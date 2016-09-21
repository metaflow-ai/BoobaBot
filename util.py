import re

# Takes a string and return a list of words
def clean_text(text):
    text = text.lower()
    text = re.sub(r'(\w)\'(\w)', r' \1\' \2', text)
    text = re.sub(r'\{.*}', '', text)
    text = re.sub(r'æ', 'ae', text)
    text = re.sub(r'œ', 'oe', text)
    text = re.sub(r'[ìíîï]', 'i', text)
    text = re.sub(r'[ýÿ]', 'y', text)
    text = re.sub(r'[òóôõö]', 'o', text)
    text = re.sub(r'[áâãä]', 'a', text)
    text = re.sub(r'ë', 'e', text)
    text = re.sub(r'ñ', 'n', text)
    text = re.sub(r'[ûü]', 'u', text)
    text = re.sub(r'\.\.\.', ' ', text)
    text = re.sub(r'([\.,?;:!"])', r' \1 ', text)
    # text = re.sub(r'`|~|!|@|#|\$|%|\^|&|\*|\(|\)|_|\+|=|<|>|\?|/|\.|,|;|:|"|\\', '', text)
    # text = re.sub(r'[`~!@#\$%\^&\*\(\)\[\]_\+=<>\?/\.,;:"\\]', '', text)
    text = re.sub(r'[^a-zA-Z0-9 àáâãäçèéêëìíîïñòóôõöùúûüýÿ\'"\.,?;:\'"!-]', '', text)
    text = re.sub(r'(x|X)\d+', '', text) # Remove x2, x3 etc.
    
    
    # text = ''.join(c for c in text if (c.isalnum()) or c == ' ')
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