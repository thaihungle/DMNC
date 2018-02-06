import tensorflow as tf
import memory

class CachedMemory(memory.Memory):
    def init_memory(self):
        """
        returns the initial values for the memory Parameters
        Returns: Tuple
        """
        print('init cached mem')

        return (
            # each sample in batch has its own version of memory
            tf.fill([self.batch_size, self.words_num, self.word_size], 1e-6),  # initial memory matrix
            tf.zeros([self.batch_size, self.words_num, ]),  # initial usage vector u
            tf.zeros([self.batch_size, self.words_num, ]),  # initial precedence vector p
            tf.zeros([self.batch_size, self.words_num, self.words_num]),  # initial link matrix L
            tf.fill([self.batch_size, self.words_num, ], 1e-6),  # initial write weighting
            tf.fill([self.batch_size, self.words_num, self.read_heads], 1e-6),  # initial read weightings
            tf.fill([self.batch_size, self.word_size, self.read_heads], 1e-6),  # initial read vectors
            tf.fill([self.batch_size, self.word_size], 1e-6),  # initial last write,
            tf.fill([self.batch_size,1], 1.0),  # initial decay prob
        )


