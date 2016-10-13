# Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, time, json, os

import tensorflow as tf
from tensorflow.models.rnn.ptb import reader

from util import word_to_id

UNKNOWN_TOKEN = '<UNK>'

class RNN(object):
    def __init__(self, config, mode='training'):
        # Training config
        self.debug = config.get('debug', False)
        self.log_dir = config.get('log_dir', 'results/rnn/' + str(int(time.time()))) 
        self.eval_freq = config.get('eval_freq', 100)
        self.save_freq = config.get('save_freq', 1000)
        self.num_epochs = config.get('num_epochs', 20)
        self.batch_size = config.get('batch_size', 32)
        self.lr = config.get('lr', 1e-3)

        # Embedding config
        self.train_glove = config.get('train_glove', False)
        self.restore_embedding = config.get('restore_embedding', True)
        if self.restore_embedding is True:
            self.glove_dir = config['glove_dir']
            self.embedding_var_name = config['embedding_var_name']
            self.embedding_chkpt = tf.train.get_checkpoint_state(self.glove_dir)
            if self.embedding_chkpt is None:
                print("Error: no saved GloVe model found in %s. Please train first" % self.glove_dir)
                sys.exit(0)
            print('Embedding will be loaded from %s' % self.embedding_chkpt.model_checkpoint_path)
        self.word_to_id_dict = config.get('word_to_id_dict', None)
        if self.word_to_id_dict is None:
            self.load_embedding_config_from_file()
        else:
            self.vocab_size = config['vocab_size']
            self.embedding_size = config['embedding_size']

        # RNN config
        if os.path.isfile(self.log_dir + '/config.json'):
            # In this case, it means that we are laoding an RNN
            self.load_rnn_config_from_file(self.log_dir + '/config.json')
        else:
            self.cell_name = config.get('cell_name', 'lstm')
            self.rnn_activation = config.get('rnn_activation', 'tanh')
            self.seq_length = config.get('seq_length', 32)
            self.state_size = config.get('state_size', 256)
            self.num_layers = config.get('num_layers', 1)
            self.tye_embedding = config.get('tye_embedding', False)
            if self.tye_embedding is True:
                print('Embedding weights will be tyed and state_size==embedding_size')
                self.state_size = self.embedding_size

        self.graph = tf.Graph()
        self.build()

    def load_rnn_config_from_file(self, config_filepath):
        print('Loading RNN config from file %s' % self.log_dir + '/config.json')
        with open(self.log_dir + '/config.json', 'w') as jsonData:
            config = json.load(jsonData)
            self.cell_name = config['cell_name']
            self.rnn_activation = config['rnn_activation']
            self.seq_length = config['seq_length']
            self.state_size = config['state_size']
            self.num_layers = config['num_layers']
            self.tye_embedding = config['tye_embedding']
            

    def load_embedding_config_from_file(self):
        print('Loading embedding config from file %s' % self.log_dir + '/config.json')
        with open(self.log_dir + '/config.json') as jsonData:
            config = json.load(jsonData)
            self.word_to_id_dict = config['word_to_id_dict']
            self.id_to_word_dict = {v: k for k, v in self.word_to_id_dict.items()}
            self.vocab_size = config['vocab_size']
            self.embedding_size = config['embedding_size']
        if not '<UNK>' in self.word_to_id_dict:
            print('Missing <UNK> word')
            sys.exit(0)

    def build(self):
        with self.graph.as_default():
            with tf.variable_scope('Placeholder'):
                self.x_plh = tf.placeholder(tf.int32, shape=[None, self.seq_length], name='Inputs_placeholder')
                self.y_plh = tf.placeholder(tf.int32, shape=[None, self.seq_length], name='Labels_placeholder')
                # self.batch_size_plh = tf.placeholder(tf.int32, shape=[1], name='Batch_size_placeholder')

            with tf.variable_scope('embedding'):
                if self.train_glove is True:
                    print('GloVe will be fine-tuned')

                self.embedding = tf.get_variable(
                    'embedding',
                    trainable=self.train_glove,
                    shape=[self.vocab_size, self.embedding_size],
                    initializer=tf.random_uniform_initializer(minval=0.0, maxval=1.0)
                )
                rnn_inputs = tf.nn.embedding_lookup(self.embedding, self.x_plh)

            # Embedding data handler
            if self.restore_embedding is True:
                self.embedding_saver = tf.train.Saver({
                    self.embedding_var_name: self.embedding
                })

            with tf.variable_scope('Encoder'):
                if self.rnn_activation == "tanh":
                    activation = tf.tanh
                elif self.rnn_activation == "relu":
                    activation = tf.nn.relu
                else:
                    raise ValueError("Activation %s not handled" % (self.rnn_activation))

                state_is_tuple = False
                if self.cell_name == 'lstm':
                    state_is_tuple = True
                    cell = tf.nn.rnn_cell.LSTMCell(self.state_size, use_peepholes=False, state_is_tuple=state_is_tuple, activation=activation)
                elif self.cell_name == 'peepholelstm':
                    state_is_tuple = True
                    cell = tf.nn.rnn_cell.LSTMCell(self.state_size, use_peepholes=True, state_is_tuple=state_is_tuple, activation=activation)
                elif self.cell_name == 'gru':
                    cell = tf.nn.rnn_cell.GRUCell(self.state_size, activation=activation)
                else:
                    raise ValueError("Cell %s not handled" % (self.cell_name))

                if self.num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers, state_is_tuple=state_is_tuple)

                self.init_state = cell.zero_state(tf.shape(rnn_inputs)[0], tf.float32)
                self.outputs, self.encoder_final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=self.init_state)

            with tf.variable_scope('Outputs'):
                if self.tye_embedding is True:
                    W_o = tf.transpose(self.embedding)
                else:
                    W_o = tf.get_variable('W_o', shape=[self.state_size, self.vocab_size])
                b_o = tf.get_variable('b_o', shape=[self.vocab_size])

                outputs = tf.reshape(self.outputs, [-1, self.state_size])
                outputs = tf.matmul(outputs, W_o) + b_o

            with tf.variable_scope('Loss'):
                y_true_reshaped = tf.reshape(self.y_plh, [-1])
                # Sparse softmax handle for us the fact that y_true is an list of indices
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, y_true_reshaped)

                self.total_loss = tf.reduce_mean(losses, name="total_loss")

                tf.scalar_summary('Total_loss', self.total_loss)

            self.train_summaries_op = tf.merge_all_summaries()

            with tf.variable_scope('Accuracy'):
                predictions = tf.cast(tf.argmax(outputs, 1, name="predictions"), tf.int32)
                correct_predictions = tf.equal(predictions, y_true_reshaped)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

                acc_summary_op = tf.scalar_summary('Accuracy', self.accuracy)

            self.dev_summaries_op = tf.merge_summary([acc_summary_op])
            self.test_summaries_op = tf.merge_summary([acc_summary_op])

            with tf.variable_scope('Prediction'):
                self.T_plh = tf.placeholder(tf.float32, shape=[], name='T_placeholder')
                self.top_k = tf.placeholder(tf.int32, shape=[], name='top_k_placeholder')

                # Determinist prediction
                preds = tf.nn.softmax(outputs)
                self.pred_topk_value, self.pred_topk = tf.nn.top_k(preds, k=self.top_k, sorted=True)

                T_preds = tf.nn.softmax(outputs / self.T_plh)
                self.random_pred = tf.multinomial(T_preds, 1)

            adam = tf.train.AdamOptimizer(self.lr)
            self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
            self.train_op = adam.minimize(self.total_loss, global_step=self.global_step)

            self.saver = tf.train.Saver()            

    def fit(self, train_data, dev_data):
        dev_iterator = reader.ptb_iterator(dev_data, self.batch_size, self.seq_length)
        x_dev_batch, y_dev_batch = next(dev_iterator)

        with tf.Session(graph=self.graph) as sess:
            sw = tf.train.SummaryWriter(self.log_dir, sess.graph)

            print('Initializing all variables')
            sess.run(tf.initialize_all_variables())

            # Restoring embedding
            if self.restore_embedding is True:
                print('restoring embedding: %s' % self.embedding_chkpt.model_checkpoint_path)
                self.embedding_saver.restore(sess, save_path=self.embedding_chkpt.model_checkpoint_path)

            for i in range(self.num_epochs):
                train_iterator = reader.ptb_iterator(train_data, self.batch_size, self.seq_length)
                for x_batch, y_batch in train_iterator:
                    _, train_summaries, total_loss, current_step = self.train_step(sess, x_batch, y_batch)
                    sw.add_summary(train_summaries, current_step)

                    if current_step % self.eval_freq == 0:
                        dev_summaries = self.dev_step(sess, x_dev_batch, y_dev_batch)
                        sw.add_summary(dev_summaries, current_step)

                    if current_step % self.save_freq == 0:
                        self.saver.save(sess, self.log_dir + '/rnn.chkp', global_step=current_step)
                epoch_acc = self.eval(sess, dev_data)
                print('Epoch: %d, Accuracy: %f' % (i + 1, epoch_acc))

            self.save(sess)

    def train_step(self, sess, x_batch, y_batch):
        to_compute = [self.train_op, self.train_summaries_op, self.total_loss, self.global_step]
        return sess.run(to_compute, feed_dict={
            self.x_plh: x_batch,
            self.y_plh: y_batch
        })

    def dev_step(self, sess, x_batch, y_batch):
        to_compute = self.dev_summaries_op
        return sess.run(to_compute, feed_dict={
            self.x_plh: x_batch,
            self.y_plh: y_batch
        })

    def eval(self, sess, test_data):    
        test_iterator = reader.ptb_iterator(test_data, self.batch_size, self.seq_length)
        nb_step = 0
        avg_acc = 0
        for x_batch, y_batch in test_iterator:
            nb_step += 1
            avg_acc += sess.run(self.accuracy, feed_dict={
                self.x_plh: x_batch,
                self.y_plh: y_batch
            })
        avg_acc /= nb_step

        return avg_acc

    def predict(self, words, T=1., random=False, sentence=False, top_k=1):
        x = [word_to_id(self.word_to_id_dict, word) for word in words]
        outputs = []

        with tf.Session(graph=self.graph) as sess:
            self.restore(sess)

            final_state = None

            if sentence is True:
                end_word = word_to_id(self.word_to_id_dict, ".")
                y = x

                max_number_of_word = 50
                i = 0
                while y[-1] != end_word and i < max_number_of_word:
                    i += 1

                    y, final_state = self.__predict_word(sess, [y], init_state=final_state, T=T, random=random)
                    outputs.append(y[-1])
            else:
                y, final_state = self.__predict_word(sess, [x], init_state=final_state, T=T, random=random, top_k=top_k)
                outputs = y
        outputs = [self.id_to_word_dict[id] for id in outputs]
        return outputs


    def __predict_word(self, sess, x, init_state=None, T=1., random=False, top_k=1):
        feed_dict = {
            self.x_plh: x,
            self.T_plh: T,
            self.top_k: top_k
        }
        if init_state != None:
            feed_dict[self.init_state] = init_state

        if random is True:
            to_compute = [self.random_pred, self.encoder_final_state]
        else:
            to_compute = [self.pred_topk, self.encoder_final_state]

        ys, final_state = sess.run(to_compute, feed_dict=feed_dict)
        return ys[-1], final_state

    def save(self, sess):
        if self.debug:
            print('Saving model to %s' % self.log_dir)
        global_step = tf.train.global_step(sess, self.global_step)
        self.saver.save(sess, self.log_dir + '/boobabot', global_step=global_step)
        config = {
            'eval_freq': self.eval_freq,
            'save_freq': self.save_freq,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,

            'train_glove': self.train_glove,
            'restore_embedding': self.restore_embedding,
            'word_to_id_dict': self.word_to_id_dict,
            'vocab_size': self.vocab_size,
            'embedding_size': self.embedding_size,

            'cell_name': self.cell_name,
            'rnn_activation': self.rnn_activation,
            'seq_length': self.seq_length,
            'state_size': self.state_size,
            'num_layers': self.num_layers,
            'tye_embedding': self.tye_embedding,
        }
        if self.restore_embedding is True:
            config['glove_dir'] = self.glove_dir
            config['embedding_var_name'] = self.embedding_var_name

        config_filepath = self.log_dir + '/config.json'
        if not os.path.isfile(config_filepath):
            with open(config_filepath, 'w') as f:
                json.dump(config, f)

    def restore(self, sess):
        print('loading model')
        checkpoint = tf.train.get_checkpoint_state(self.log_dir)
        if checkpoint is None:
            print("Error: no saved model found in %s. Please train first" % self.log_dir)
            sys.exit(0)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)
