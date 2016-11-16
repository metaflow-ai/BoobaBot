import os, argparse, json

from models.rnn import RNN
from util import clean_text

dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--debug", nargs="?", const=True, default=False, type=bool, help="debug mode (default: %(default)s)")
parser.add_argument("--model_dir", default="rnn", type=str, help="glove dir (default: %(default)s)")
parser.add_argument("--inputs", default="Un iench", type=str, help="Choose the beginning of the sentence")
parser.add_argument("--random", nargs="?", const=True, default=False, type=bool, help="Predict using temperature (default: %(default)s)")
parser.add_argument("--temperature", default=1., type=float, help="The temperature for prediction (default: %(default)s)")
parser.add_argument("--top_k", default=1, type=int, help="Return the top K prediction (default: %(default)s)")
parser.add_argument("--nb_word", default=-1, type=int, help="How many words should it return (default: %(default)s, -1: no limit)")
parser.add_argument("--nb_sentence", default=-1, type=int, help="How many lines should it return (default: %(default)s, -1: no limit)")
parser.add_argument("--nb_para", default=1, type=int, help="How many para should it return (default: %(default)s, -1: no limit)")
parser.add_argument('--use_server', nargs="?", const=True, default=False, type=bool, help='Should use the Server architecture')
args = parser.parse_args()

results_dir = dir + '/results'
rnn_dir = dir + '/' + args.model_dir

config = vars(args)
config['log_dir'] = rnn_dir
config['restore_embedding'] = False
config['seq_length'] = None
input_words = clean_text(config['inputs'])
if args.use_server is True:
    with open('clusterSpec.json') as f:
        clusterSpec = json.load(f)
    config['target'] = 'grpc://' + clusterSpec['server'][0]
    pass

rnn = RNN(config)
y = rnn.predict(input_words, config)
print('__BBB_START__') 
json = json.dumps({
    'config': {
        'inputs': args.inputs,
        'random': args.random,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'nb_word': args.nb_word,
        'nb_sentence': args.nb_sentence,
        'nb_para': args.nb_para,
    },
    'output': ' '.join(y)
})
print(json)
print('__BBB_END__')