import json
import tensorflow as tf

with open('clusterSpec.json') as f:
    clusterSpec = json.load(f)

cluster = tf.train.ClusterSpec(clusterSpec)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess_config = tf.ConfigProto(gpu_options=gpu_options)

# Start the server
server = tf.train.Server(cluster, config=sess_config)

# Hang on bro
server.join()