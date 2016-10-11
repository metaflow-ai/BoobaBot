import time, os, json, argparse, sys

from util import load_corpus_as_sets
from models.rnn import RNN

dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--debug", nargs="?", const=True, default=False, type=bool, help="debug mode (default: %(default)s)")
parser.add_argument("--model_dir", default="pretrain", type=str, help="Choose the glove initialization (default: %(default)s)")
parser.add_argument("--input", default="glove", type=str, help="glove dir (default: %(default)s)")
parser.add_argument("--temperature", default=1, type=int, help="temperature for prediction (default: %(default)s)")
parser.add_argument("--random", nargs="?", const=True, default=False, type=bool, help="random mode (Use it with the temperature param default: %(default)s)")
parser.add_argument("--sentence", nargs="?", const=True, default=False, type=bool, help="debug mode (default: %(default)s)")
args = parser.parse_args()

results_dir = dir + '/results'
glove_dir = results_dir + '/' + args.glove_dir
rnn_log_dir = results_dir + '/rnn/' + str(int(time.time()))

with open(glove_dir + '/data.json') as jsonData:
    rawData = json.load(jsonData)

config = dict(rawData['config'])
config.update(vars(args))
config['word_to_id_dict'] = rawData['word_to_id_dict']
config['glove_dir'] = glove_dir

config['restore_embedding'] = False

print('Loading corpus as sets')
if args.debug is True:
    fullpath = dir + '/crawler/data/test_results.txt'
else:
    fullpath = dir + '/crawler/data/results.txt'
train_data, dev_data, test_data = load_corpus_as_sets(fullpath, rawData['word_to_id_dict'])

print('Building graph')
model = RNN(config)

outputs = model.eval(input, dev_data, args.temperature, args.random, args.sentence)
print(' '.join(inputs) + ' ... ' + ' '.join(outputs))

