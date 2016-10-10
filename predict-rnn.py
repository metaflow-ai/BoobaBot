import tensorflow as tf
import time, os, json, argparse, sys

from models.rnn import RNN
from util import word_to_id

dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--sentence", default="Sale", type=str, help="Choose the beginning of the sentence")
parser.add_argument("--glove_dir", default="glove", type=str, help="glove dir (default: %(default)s)")
parser.add_argument("--model_dir", default="rnn", type=str, help="glove dir (default: %(default)s)")
args = parser.parse_args()

results_dir = dir + '/results'
glove_dir = results_dir + '/' + args.glove_dir
rnn_dir = results_dir + '/' + args.model_dir

with open(glove_dir + '/data.json') as jsonData:
    rawData = json.load(jsonData)

config = dict(rawData['config'])
config.update(vars(args))
config['glove_dir'] = glove_dir
config['log_dir'] = rnn_dir
# config['restore_embedding'] = False

config['word_to_id_dict'] = rawData['word_to_id_dict']
config['seq_length'] = None

rnn = RNN(config)

sentence = [word_to_id(config['word_to_id_dict'], word) for word in config['sentence'].split(' ')]


with tf.Session(graph=rnn.graph) as sess:
    sess.run(tf.initialize_all_variables())

    rnn.load(sess)
    y = rnn.predict([sentence])
    print(y)