import tensorflow as tf
import time, os, json, argparse, sys

from models.rnn import RNN
from util import word_to_id, clean_text

dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--debug", nargs="?", const=True, default=False, type=bool, help="debug mode (default: %(default)s)")
parser.add_argument("--model_dir", default="rnn", type=str, help="glove dir (default: %(default)s)")
parser.add_argument("--inputs", default="Sale my", type=str, help="Choose the beginning of the sentence")
parser.add_argument("--random", nargs="?", const=True, default=False, type=bool, help="Predict using temperature (default: %(default)s)")
parser.add_argument("--temperature", default=1., type=float, help="The temperature for prediction (default: %(default)s)")
parser.add_argument("--sentence", nargs="?", const=True, default=False, type=bool, help="Should predict the full sentence, until the <END> token (default: %(default)s)")
parser.add_argument("--topk", default=1, type=int, help="Return the top K prediction (default: %(default)s)")
args = parser.parse_args()

results_dir = dir + '/results'
rnn_dir = dir + '/' + args.model_dir

config = vars(args)
config['log_dir'] = rnn_dir
config['restore_embedding'] = False
config['seq_length'] = None
input_words = clean_text(config['inputs'])

rnn = RNN(config)
y = rnn.predict(input_words, config['temperature'], config['random'], config['sentence'], config['topk'])
print(config['inputs'] + ' ... ' + ' '.join(y))