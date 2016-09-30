import time, os, json, argparse, sys

from util import load_corpus_as_sets
from models.rnn import RNN

dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--debug", nargs="?", const=True, default=False, type=bool, help="debug mode (default: %(default)s)")
parser.add_argument("--init_glove", default="pretrain", type=str, help="Choose the glove initialization (default: %(default)s)")
parser.add_argument("--glove_dir", default="glove", type=str, help="glove dir (default: %(default)s)")
parser.add_argument("--train_glove", nargs="?", const=True, default=False, type=bool, help="Are we finetuning/training GloVe embedding (default: %(default)s)")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size (default: %(default)s)")
parser.add_argument("--seq_length", default=32, type=int, help="RNN sequence length (default: %(default)s)")
parser.add_argument("--state_size", default=256, type=int, help="RNN state size (default: %(default)s)")
parser.add_argument("--num_layers", default=1, type=int, help="How deep is the RNN (default: %(default)s)")
parser.add_argument("--num_epochs", default=50, type=int, help="How many epochs should we train the RNN (default: %(default)s)")
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
if not '<UNK>' in config['word_to_id_dict']:
    print('Missing <UNK> word')
    sys.exit(0)

print('Loading corpus as sets')
if args.debug is True:
    fullpath = dir + '/crawler/data/test_results.txt'
else:
    fullpath = dir + '/crawler/data/results.txt'
train_data, dev_data, test_data = load_corpus_as_sets(fullpath, rawData['word_to_id_dict'])

print('Building graph')
model = RNN(config)

print('Training model')
model.fit(train_data, dev_data)

