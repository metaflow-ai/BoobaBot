######################################################
#                                                    #
# !!! This script needs at least tensorflow 0.12 !!! # 
#                                                    #
######################################################

import os, argparse, json, time
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from t_sne import load_embedding

dir = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = dir + '/results/embedding_projector/' + str(int(time.time()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_dir", default="results/glove", type=str, help="glove dir (default: %(default)s)")
    args = parser.parse_args()

    glove_dir = dir + '/' + args.glove_dir
    glove_name = args.glove_dir.split('/')[-1]
    if glove_name == '':
        glove_name = args.glove_dir.split('/')[-2]

    with open(glove_dir + '/config.json') as jsonData:
        rawData = json.load(jsonData)
        config = dict(rawData['config'])
        config['glove_dir'] = glove_dir

    embedding_value = load_embedding(config)
    embedding_var = tf.Variable(embedding_value, "embedding_var")

    sw = tf.train.SummaryWriter(LOG_DIR)
    projector_config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = projector_config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, os.path.join(LOG_DIR, "embedding_var.ckpt"))

        projector.visualize_embeddings(sw, projector_config)