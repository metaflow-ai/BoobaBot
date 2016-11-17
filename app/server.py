import os, sys, json, argparse

from flask import Flask, request
from flask_cors import CORS, cross_origin

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

from models.rnn import RNN
from util import clean_text

app = Flask(__name__)
cors = CORS(app)

@app.route("/api/predict", methods=['POST'])
def predict():
    params = json.loads(request.data.decode("utf-8"))
    inputs = params['inputs']
    random = True if params.get('random', False) == True or params.get('random', False) == 'true' else False
    temperature = float(params.get('temperature', 1.))
    top_k = int(params.get('topk', 1))
    number = int(params.get('number', 1))
    kind = params.get('kind', 'word')
    if kind == 'para':
        nb_para = number
        nb_sentence = -1
        nb_word = -1
    elif kind == 'sentence':
        nb_para = 1
        nb_sentence = number
        nb_word = -1
    else:
        nb_para = 1
        nb_sentence = -1
        nb_word = number
    config = {
        'inputs': inputs,
        'random': random,
        'temperature': temperature,
        'top_k': top_k,
        'nb_word': nb_word,
        'nb_sentence': nb_sentence,
        'nb_para': nb_para,
    }

    input_words = clean_text(inputs)
    if len(input_words) == 0:
        input_words.append("")
    y = rnn.predict(input_words, config)

    return json.dumps({
        'config': config,
        'output': ' '.join(y)
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", nargs="?", const=True, default=False, type=bool, help="debug mode (default: %(default)s)")
    parser.add_argument("--model_dir", default="/../results/rnn/1476859465", type=str, help="glove dir (default: %(default)s)")
    args = parser.parse_args()

    rnn_dir = dir + '/' + args.model_dir  
    config_rnn = {}
    config_rnn['log_dir'] = rnn_dir
    config_rnn['restore_embedding'] = False
    config_rnn['seq_length'] = None

    print('Loading config')
    rnn = RNN(config_rnn)
    print('Starting Session')
    rnn.start_session()

    app.run()
