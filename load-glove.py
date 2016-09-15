import time, os

import tensorflow as tf

with open('crawler/data/results.txt', 'r') as f:
    corpus = f.readlines()
    for i,line in enumerate(corpus):
        corpus[i] = line.strip().split(' ')

vocab_size = 27347
embedding_size = 3
with tf.variable_scope('embedding', initializer=tf.random_uniform_initializer(-1.0, 1.0)):
    focal_embeddings = tf.get_variable(name="focal_embeddings", trainable=False, shape=[vocab_size, embedding_size])
    # focal_biases = tf.get_variable(name='focal_biases', trainable=False, shape=[vocab_size])
    context_embeddings = tf.get_variable(name="context_embeddings", trainable=False, shape=[vocab_size, embedding_size])
    # context_biases = tf.get_variable(name="context_biases", trainable=False, shape=[vocab_size])

    embedding_saver = tf.train.Saver({
        "focal_embeddings": focal_embeddings,
        # "focal_biases": focal_biases,
        "context_embeddings": context_embeddings,
        # "context_biases": context_biases,
    })

with tf.Session() as sess:    
    checkpoint_file = 'results/1473925094/booba-embedding-512'
    embedding_saver.restore(sess, checkpoint_file)

    combined_embeddings = tf.add(focal_embeddings, context_embeddings, name="combined_embeddings")
    print(combined_embeddings.eval())
    
