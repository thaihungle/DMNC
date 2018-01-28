import tensorflow as tf
from controller import BaseController
import numpy as np

class FeedforwardController(BaseController):
    def network_vars(self):
        initial_std = lambda in_nodes: np.min([1e-2, np.sqrt(2.0 / in_nodes)])
        input_ = int(self.nn_input_size)

        self.W1 = tf.Variable(tf.truncated_normal([input_, self.hidden_dim], stddev=initial_std(input_)), name='layer1_W')
        self.W2 = tf.Variable(tf.truncated_normal([self.hidden_dim, self.hidden_dim*2],
                                                  stddev=initial_std(self.hidden_dim)), name='layer2_W')
        self.b1 = tf.Variable(tf.zeros([self.hidden_dim]), name='layer1_b')
        self.b2 = tf.Variable(tf.zeros([self.hidden_dim*2]), name='layer2_b')

    def network_op(self, X):
        l1_output = tf.matmul(X, self.W1) + self.b1
        l1_activation = tf.nn.relu(l1_output)

        l2_output = tf.matmul(l1_activation, self.W2) + self.b2
        l2_activation = tf.nn.relu(l2_output)

        return l2_activation

    def initials(self):
        initial_std = lambda in_nodes: np.min([1e-2, np.sqrt(2.0 / in_nodes)])

        # defining internal weights of the controller
        self.interface_weights = tf.Variable(
            tf.truncated_normal([self.nn_output_size, self.interface_vector_size],
                                stddev=initial_std(self.nn_output_size)),
            name='interface_weights'
        )
        self.nn_output_weights = tf.Variable(
            tf.truncated_normal([self.nn_output_size, self.output_size], stddev=initial_std(self.nn_output_size)),
            name='nn_output_weights'
        )
        self.mem_output_weights = tf.Variable(
            tf.truncated_normal([self.word_size * self.read_heads, self.output_size],
                                stddev=initial_std(self.word_size * self.read_heads)),
            name='mem_output_weights')