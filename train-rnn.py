import time, os, json

import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))
main_dir = dir + 'results/baseline'

with open(main_dir + '/data.json') as jsonData:
    rawData = json.load(jsonData)

vocab_size = rawData['vocab_size']
embedding_size = rawData['learning_config']['embedding_size']

# Load with tensorflow not needed
# with tf.variable_scope('embedding', initializer=tf.random_uniform_initializer(-1.0, 1.0)):
#     focal_embeddings = tf.get_variable(name="focal_embeddings", trainable=False, shape=[vocab_size, embedding_size])
#     # focal_biases = tf.get_variable(name='focal_biases', trainable=False, shape=[vocab_size])
#     context_embeddings = tf.get_variable(name="context_embeddings", trainable=False, shape=[vocab_size, embedding_size])
#     # context_biases = tf.get_variable(name="context_biases", trainable=False, shape=[vocab_size])

#     embedding_saver = tf.train.Saver({
#         "focal_embeddings": focal_embeddings,
#         # "focal_biases": focal_biases,
#         "context_embeddings": context_embeddings,
#         # "context_biases": context_biases,
#     })

# with tf.Session() as sess:    
#     checkpoint_file = main_dir + '/booba-embedding'
#     embedding_saver.restore(sess, checkpoint_file)

#     combined_embeddings = tf.add(focal_embeddings, context_embeddings, name="combined_embeddings")
#     embedding = combined_embeddings.eval()
word_to_id = rawData['word_to_id']
embed = rawData['embed']

# Hyper param
batch_size = 32
num_steps = 10
state_size = 256

inputs = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name='Inputs_placeholder')
labels = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name='Labels_placeholder')

with tf.variable_scope('embedding'):
    embedding = tf.get_variable('embedding', trainable=False, initializer=tf.constant_initializer(embed))

    rnn_inputs = words = tf.nn.embedding_lookup(embedding, inputs)

with tf.variable_scope('LSTM'):
    state_is_tuple = True
    cell = tf.nn.rnn_cell.LSTMCell(state_size, use_peepholes=True, state_is_tuple=state_is_tuple)
    num_layers = 2
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=state_is_tuple)

    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

with tf.variable_scope('Softmax'):
    W_s = tf.get_variable('W_s', shape=[state_size, vocab_size])
    b_s = tf.get_variable('b_s', shape=[vocab_size])

    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    labels_reshaped = tf.reshape(labels, [-1])
    logits = tf.nn.Softmax(tf.matmul(rnn_outputs, W_s) + b_s)

    losses = tf.nn.softmax_cross_entropy_with_logits(logits, labels_reshaped)

    total_loss = tf.reduce_mean(losses)
    tf.scalar_summary('total_loss', total_loss)

adam = tf.train.AdamOptimizer(1e-3)
train_op = adam.minimize(total_loss)