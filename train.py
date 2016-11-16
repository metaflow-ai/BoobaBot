import os, json, argparse

from util import load_corpus_as_sets
from models.rnn import RNN

dir = os.path.dirname(os.path.realpath(__file__))

def main(args):
    glove_dir = args.glove_dir

    with open(glove_dir + '/config.json') as jsonData:
        rawData = json.load(jsonData)

    # merge the two configuration
    config = dict(rawData['config'])
    config.update(vars(args)) 
    config['word_to_id_dict'] = rawData['word_to_id_dict']
    config['glove_dir'] = glove_dir

    if args.debug is True:
        print('Loading corpus as sets')
    textfile = args.textfile
    train_data, dev_data, test_data = load_corpus_as_sets(textfile, rawData['word_to_id_dict'])

    if args.debug is True:
        print('Building graph')
    model = RNN(config)

    if args.debug is True:
        print('Training model')
    model.fit(train_data, dev_data)

    return model.log_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training config
    parser.add_argument("--debug", nargs="?", const=True, default=False, type=bool, help="debug mode (default: %(default)s)")
    parser.add_argument("--profiling", nargs="?", const=True, default=False, type=bool, help="debug mode (default: %(default)s)")
    parser.add_argument("--num_epochs", default=20, type=int, help="How many epochs should we train the RNN (default: %(default)s)")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size (default: %(default)s)")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate for the adam optimizer (default: %(default)s)")
    # Embedding config
    parser.add_argument("--textfile", default=dir + '/crawler/data/results.txt', type=str, help="The textfile used to train GloVe (default: %(default)s)")
    parser.add_argument("--glove_dir", default=dir + "/results/glove", type=str, help="glove dir (default: %(default)s)")
    parser.add_argument("--train_glove", nargs="?", const=True, default=False, type=bool, help="Are we finetuning/training GloVe embedding (default: %(default)s)")
    # RNN config
    parser.add_argument("--cell_name", default="lstm", type=str, help="Cell architecture (default: %(default)s)")
    parser.add_argument("--rnn_activation", default="tanh", type=str, help="RNN activation function name (default: %(default)s)")
    parser.add_argument("--seq_length", default=32, type=int, help="RNN sequence length (default: %(default)s)")
    parser.add_argument("--state_size", default=256, type=int, help="RNN state size (default: %(default)s)")
    parser.add_argument("--num_layers", default=1, type=int, help="How deep is the RNN (default: %(default)s)")
    parser.add_argument("--tye_embedding", nargs="?", const=True, default=False, type=bool, help="Tye word embedding weights to compute the outputs value (default: %(default)s)")

    args = parser.parse_args()
    main(args)


