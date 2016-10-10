# Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
from tensorflow.models.rnn.ptb import reader

from util import word_to_id

class RNN(object):
    def __init__(self, config):
        self.debug = config.get('debug', False)
        self.log_dir = config.get('log_dir', 'results/rnn')
        self.eval_freq = config.get('eval_freq', 100)
        self.save_freq = config.get('save_freq', 100)

        self.num_epochs = config.get('num_epochs', 20)
        self.batch_size = config.get('batch_size', 32)
        self.lr = config.get('lr', 1e-3)

        self.train_glove = config.get('train_glove', False)
        self.restore_embedding = config.get('restore_embedding', True)
        if self.restore_embedding is True:
            self.glove_dir = config['glove_dir']
            self.embedding_var_name = config['embedding_var_name']
            self.embedding_chkpt_file = config['embedding_chkpt_file']
        self.word_to_id_dict = config['word_to_id_dict']

        self.seq_length = config.get('seq_length', 32)
        self.state_size = config.get('state_size', 256)
        self.vocab_size = config['vocab_size']
        self.embedding_size = config['embedding_size']

        self.num_layers = config.get('num_layers', 1)

        self.graph = tf.Graph()
        self.build()


    def build(self):
        with self.graph.as_default():
            with tf.variable_scope('Placeholder'):
                self.x_plh = tf.placeholder(tf.int32, shape=[None, self.seq_length], name='Inputs_placeholder')
                self.y_plh = tf.placeholder(tf.int32, shape=[None, self.seq_length], name='Labels_placeholder')
                # self.batch_size_plh = tf.placeholder(tf.int32, shape=[1], name='Batch_size_placeholder')

            with tf.variable_scope('embedding'):
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
                state_is_tuple = True
                cell = tf.nn.rnn_cell.LSTMCell(self.state_size, use_peepholes=True, state_is_tuple=state_is_tuple)
                if self.num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers, state_is_tuple=state_is_tuple)
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers, state_is_tuple=state_is_tuple)

                init_state = cell.zero_state(self.batch_size, tf.float32)
                outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

            with tf.variable_scope('Outputs'):
                W_s = tf.get_variable('W_s', shape=[self.state_size, self.vocab_size])
                b_s = tf.get_variable('b_s', shape=[self.vocab_size])

                outputs = tf.reshape(outputs, [-1, self.state_size])
                outputs = tf.matmul(outputs, W_s) + b_s

            with tf.variable_scope('Loss'):
                y_true_reshaped = tf.reshape(self.y_plh, [-1])
                # Sparse softmax handle for us the fact that y_true is an list of indices
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, y_true_reshaped)

                total_loss = tf.reduce_mean(losses, name="total_loss")

                tf.scalar_summary('Total_loss', total_loss)

            self.train_summaries_op = tf.merge_all_summaries()

            with tf.variable_scope('Accuracy'):
                predictions = tf.cast(tf.argmax(outputs, 1, name="predictions"), tf.int32)
                correct_predictions = tf.equal(predictions, y_true_reshaped)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

                acc_summary_op = tf.scalar_summary('Accuracy', self.accuracy)

            self.dev_summaries_op = tf.merge_summary([acc_summary_op])
            self.test_summaries_op = tf.merge_summary([acc_summary_op])

            adam = tf.train.AdamOptimizer(self.lr)
            self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
            self.train_op = adam.minimize(total_loss, global_step=self.global_step)

            self.saver = tf.train.Saver()


            with tf.variable_scope('Prediction'):
                self.temp_plh = tf.placeholder(tf.float32, shape=[1], name='temp_placeholder')
                # Determinist prediction
                preds = tf.nn.softmax(outputs)
                self.pred = tf.argmax(preds, 1)

                preds = tf.nn.softmax(outputs / self.temp_plh)
                self.hot_pred = tf.multinomial(preds, 1)

    def fit(self, train_data, dev_data):
        dev_iterator = reader.ptb_iterator(dev_data, self.batch_size, self.seq_length)
        x_dev_batch, y_dev_batch = next(dev_iterator)

        with tf.Session(graph=self.graph) as sess:
            sw = tf.train.SummaryWriter(self.log_dir, sess.graph)

            print('Initializing all variables')
            sess.run(tf.initialize_all_variables())

            # Restoring embedding
            if self.restore_embedding is True:
                embedding_fullpath = self.glove_dir + '/' + self.embedding_chkpt_file
                print('restoring embedding: %s' % embedding_fullpath)
                self.embedding_saver.restore(sess, save_path=embedding_fullpath)

            for i in range(self.num_epochs):
                train_iterator = reader.ptb_iterator(train_data, self.batch_size, self.seq_length)
                for x_batch, y_batch in train_iterator:
                    train_summaries = self.train_step(sess, x_batch, y_batch)

                    current_step = tf.train.global_step(sess, self.global_step)
                    print(current_step)
                    sw.add_summary(train_summaries, current_step)

                    if current_step % self.eval_freq == 0:
                        dev_summaries = self.dev_step(sess, x_dev_batch, y_dev_batch)
                        sw.add_summary(dev_summaries, current_step)

                    if current_step % self.save_freq == 0:
                        self.saver.save(sess, self.log_dir + '/rnn.chkp', global_step=current_step)
                epoch_acc = self.eval(dev_data, sess)
                print('Epoch: %d, Accuracy: %f' % (i + 1, epoch_acc))

            self.save(sess)

    def train_step(self, sess, x_batch, y_batch):
        _, summaries = sess.run([self.train_op, self.train_summaries_op], feed_dict={
            self.x_plh: x_batch,
            self.y_plh: y_batch
        })
        return summaries

    def dev_step(self, sess, x_batch, y_batch):
        acc_summary = sess.run(self.dev_summaries_op, feed_dict={
            self.x_plh: x_batch,
            self.y_plh: y_batch
        })
        return acc_summary

    def eval(self, test_data, sess=None):
        if sess is None:
            sess = tf.Session(graph=self.graph)
            self.saver.restore(self.log_dir + '/rnn.chkp')

        test_iterator = reader.ptb_iterator(test_data, self.batch_size, self.seq_length)
        nb_step = 0
        acc = 0
        for x_batch, y_batch in test_iterator:
            nb_step += 1
            acc += sess.run(self.accuracy, feed_dict={
                self.x_plh: x_batch,
                self.y_plh: y_batch
            })
        acc /= nb_step
        return acc

    def predict(self, x, temperature=1, random=False, sentence=False):
        with tf.Session(graph=self.graph) as sess:
            if sentence is True:
                end_word = word_to_id(self.word_to_id_dict, ".")
                y = None
                while y != end_word:
                    y = self.__predict_word(sess, x, temperature, random)
                    x += y
                return x
            else:
                y = self.__predict_word(sess, x, temperature, random)
                return y


    def __predict_word(self, sess, x, temperature=1, random=False):
        if random is True:
            y = sess.run([self.hot_pred], feed_dict={
                self.x_plh: x,
                self.temp_plh: [temperature]
            })
        else:
            y = sess.run([self.pred], feed_dict={
                self.x_plh: x,
                self.temp_plh: [temperature]
            })
        return y

    def save(self, sess):
        print('Saving model to %s' % self.log_dir)
        self.saver.save(sess, self.log_dir)

    def load(self, sess):
        print('loading model')
        checkpoint = tf.train.get_checkpoint_state(self.log_dir)
        if checkpoint is None:
            print("Error: no saved model found in %s. Please train first" % self.log_dir)
            sys.exit(0)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)
