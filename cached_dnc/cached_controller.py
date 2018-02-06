import tensorflow as tf
import recurrent_controller
import feedforward_controller

class CachedFWController(feedforward_controller.FeedforwardController):

    def __init__(self, input_size, output_size, memory_read_heads, memory_word_size, batch_size=1,
                 use_mem=True, hidden_dim=256, is_two_mem=False):
        """
           constructs a controller as described in the DNC paper:
           http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

           Parameters:
           ----------
           input_size: int
               the size of the data input vector
           output_size: int
               the size of the data output vector
           memory_read_heads: int
               the number of read heads in the associated external memory
           memory_word_size: int
               the size of the word in the associated external memory
           batch_size: int
               the size of the input data batch [optional, usually set by the DNC object]
           """
        self.use_mem = use_mem
        self.input_size = input_size
        self.output_size = output_size
        self.read_heads = memory_read_heads  # in dnc there are many read head but one write head
        self.word_size = memory_word_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        # indicates if the internal neural network is recurrent
        # by the existence of recurrent_update and get_state methods
        # subclass should implement these methods if it is rnn based controller
        has_recurrent_update = callable(getattr(self, 'update_state', None))
        has_get_state = callable(getattr(self, 'get_state', None))
        self.has_recurrent_nn = has_recurrent_update and has_get_state

        # the actual size of the neural network input after flatenning and
        # concatenating the input vector with the previously read vctors from memory
        if use_mem:
            if is_two_mem:
                self.nn_input_size = self.word_size * self.read_heads * 2 + self.input_size
            else:
                self.nn_input_size = self.word_size * self.read_heads + self.input_size
        else:
            self.nn_input_size = self.input_size

        self.interface_vector_size = self.word_size * self.read_heads  # R read keys
        self.interface_vector_size += 3 * self.word_size  # 1 write key, 1 erase, 1 content
        self.interface_vector_size += 5 * self.read_heads  # R read key strengths, R free gates, 3xR read modes (each mode for each read has 3 values)
        self.interface_vector_size += 3  # 1 write strength, 1 allocation gate, 1 write gate
        self.interface_vector_size += self.word_size  # spare for gate cache

        self.interface_weights = self.nn_output_weights = self.mem_output_weights = None
        self.is_two_mem = is_two_mem

        # define network vars
        with tf.name_scope("controller"):
            self.network_vars()

            self.nn_output_size = None  # not yet defined in the general scope --> output of the controller not of the whole
            with tf.variable_scope("shape_inference"):
                # depend on model type --> seperate variable scope
                self.nn_output_size = self.get_nn_output_size()

            self.initials()




    def parse_interface_vector(self, interface_vector):
        """
        pasres the flat interface_vector into its various components with their
        correct shapes

        Parameters:
        ----------
        interface_vector: Tensor (batch_size, interface_vector_size)
            the flattened inetrface vector to be parsed

        Returns: dict
            a dictionary with the components of the interface_vector parsed
        """

        parsed = {}

        r_keys_end = self.word_size * self.read_heads
        r_strengths_end = r_keys_end + self.read_heads
        w_key_end = r_strengths_end + self.word_size
        erase_end = w_key_end + 1 + self.word_size
        write_end = erase_end + self.word_size
        free_end = write_end + self.read_heads

        r_keys_shape = (-1, self.word_size, self.read_heads)
        r_strengths_shape = (-1, self.read_heads)
        w_key_shape = (-1, self.word_size, 1)
        write_shape = erase_shape = cache_shape = (-1, self.word_size)
        free_shape = (-1, self.read_heads)
        modes_shape = (-1, 3, self.read_heads)

        # parsing the vector into its individual components
        parsed['read_keys'] = tf.reshape(interface_vector[:, :r_keys_end], r_keys_shape) #batch x N x R
        parsed['read_strengths'] = tf.reshape(interface_vector[:, r_keys_end:r_strengths_end], r_strengths_shape) #batch x R
        parsed['write_key'] = tf.reshape(interface_vector[:, r_strengths_end:w_key_end], w_key_shape) #batch x N x 1 --> share similarity function with read
        parsed['write_strength'] = tf.reshape(interface_vector[:, w_key_end], (-1, 1)) # batch x 1
        parsed['erase_vector'] = tf.reshape(interface_vector[:, w_key_end + 1:erase_end], erase_shape) #batch x N
        parsed['write_vector'] = tf.reshape(interface_vector[:, erase_end:write_end], write_shape)# batch x N
        parsed['free_gates'] = tf.reshape(interface_vector[:, write_end:free_end], free_shape)# batch x R
        parsed['allocation_gate'] = tf.expand_dims(interface_vector[:, free_end], 1)# batch x 1
        parsed['write_gate'] = tf.expand_dims(interface_vector[:, free_end + 1], 1)# batch x 1
        parsed['read_modes'] = tf.reshape(interface_vector[:, free_end + 2:free_end + 5], modes_shape)# batch x 3 x R
        parsed['cache_gate'] = tf.reshape(interface_vector[:, free_end + 5:], cache_shape)  # batch x N

        # transforming the components to ensure they're in the right ranges
        parsed['read_strengths'] = 1 + tf.nn.softplus(parsed['read_strengths'])
        parsed['write_strength'] = 1 + tf.nn.softplus(parsed['write_strength'])
        parsed['erase_vector'] = tf.nn.sigmoid(parsed['erase_vector'])
        parsed['free_gates'] = tf.nn.sigmoid(parsed['free_gates'])
        parsed['allocation_gate'] = tf.nn.sigmoid(parsed['allocation_gate'])
        parsed['write_gate'] = tf.nn.sigmoid(parsed['write_gate'])
        parsed['read_modes'] = tf.nn.softmax(parsed['read_modes'], 1)
        parsed['cache_gate'] = tf.nn.sigmoid(parsed['cache_gate'])

        return parsed # dict of tensors


class CachedLSTMController(recurrent_controller.StatelessRecurrentController):

    def __init__(self, input_size, output_size, memory_read_heads, memory_word_size, batch_size=1,
                 use_mem=True, hidden_dim=256, is_two_mem=False):
        """
           constructs a controller as described in the DNC paper:
           http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

           Parameters:
           ----------
           input_size: int
               the size of the data input vector
           output_size: int
               the size of the data output vector
           memory_read_heads: int
               the number of read heads in the associated external memory
           memory_word_size: int
               the size of the word in the associated external memory
           batch_size: int
               the size of the input data batch [optional, usually set by the DNC object]
           """
        self.use_mem = use_mem
        self.input_size = input_size
        self.output_size = output_size
        self.read_heads = memory_read_heads  # in dnc there are many read head but one write head
        self.word_size = memory_word_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        # indicates if the internal neural network is recurrent
        # by the existence of recurrent_update and get_state methods
        # subclass should implement these methods if it is rnn based controller
        has_recurrent_update = callable(getattr(self, 'update_state', None))
        has_get_state = callable(getattr(self, 'get_state', None))
        self.has_recurrent_nn = has_recurrent_update and has_get_state

        # the actual size of the neural network input after flatenning and
        # concatenating the input vector with the previously read vctors from memory
        if use_mem:
            if is_two_mem:
                self.nn_input_size = self.word_size * self.read_heads * 2 + self.input_size
            else:
                self.nn_input_size = self.word_size * self.read_heads + self.input_size
        else:
            self.nn_input_size = self.input_size

        self.interface_vector_size = self.word_size * self.read_heads  # R read keys
        self.interface_vector_size += 3 * self.word_size  # 1 write key, 1 erase, 1 content
        self.interface_vector_size += 5 * self.read_heads  # R read key strengths, R free gates, 3xR read modes (each mode for each read has 3 values)
        self.interface_vector_size += 3  # 1 write strength, 1 allocation gate, 1 write gate
        self.interface_vector_size += self.word_size  # spare for gate cache

        self.interface_weights = self.nn_output_weights = self.mem_output_weights = None
        self.is_two_mem = is_two_mem

        # define network vars
        with tf.name_scope("controller"):
            self.network_vars()

            self.nn_output_size = None  # not yet defined in the general scope --> output of the controller not of the whole
            with tf.variable_scope("shape_inference"):
                # depend on model type --> seperate variable scope
                self.nn_output_size = self.get_nn_output_size()

            self.initials()




    def parse_interface_vector(self, interface_vector):
        """
        pasres the flat interface_vector into its various components with their
        correct shapes

        Parameters:
        ----------
        interface_vector: Tensor (batch_size, interface_vector_size)
            the flattened inetrface vector to be parsed

        Returns: dict
            a dictionary with the components of the interface_vector parsed
        """

        parsed = {}

        r_keys_end = self.word_size * self.read_heads
        r_strengths_end = r_keys_end + self.read_heads
        w_key_end = r_strengths_end + self.word_size
        erase_end = w_key_end + 1 + self.word_size
        write_end = erase_end + self.word_size
        free_end = write_end + self.read_heads

        r_keys_shape = (-1, self.word_size, self.read_heads)
        r_strengths_shape = (-1, self.read_heads)
        w_key_shape = (-1, self.word_size, 1)
        write_shape = erase_shape = cache_shape = (-1, self.word_size)
        free_shape = (-1, self.read_heads)
        modes_shape = (-1, 3, self.read_heads)

        # parsing the vector into its individual components
        parsed['read_keys'] = tf.reshape(interface_vector[:, :r_keys_end], r_keys_shape) #batch x N x R
        parsed['read_strengths'] = tf.reshape(interface_vector[:, r_keys_end:r_strengths_end], r_strengths_shape) #batch x R
        parsed['write_key'] = tf.reshape(interface_vector[:, r_strengths_end:w_key_end], w_key_shape) #batch x N x 1 --> share similarity function with read
        parsed['write_strength'] = tf.reshape(interface_vector[:, w_key_end], (-1, 1)) # batch x 1
        parsed['erase_vector'] = tf.reshape(interface_vector[:, w_key_end + 1:erase_end], erase_shape) #batch x N
        parsed['write_vector'] = tf.reshape(interface_vector[:, erase_end:write_end], write_shape)# batch x N
        parsed['free_gates'] = tf.reshape(interface_vector[:, write_end:free_end], free_shape)# batch x R
        parsed['allocation_gate'] = tf.expand_dims(interface_vector[:, free_end], 1)# batch x 1
        parsed['write_gate'] = tf.expand_dims(interface_vector[:, free_end + 1], 1)# batch x 1
        parsed['read_modes'] = tf.reshape(interface_vector[:, free_end + 2:free_end + 2+ 3*self.read_heads],
                                          modes_shape)# batch x 3 x R
        parsed['cache_gate'] = tf.reshape(interface_vector[:, free_end + 2+ 3*self.read_heads:], cache_shape)  # batch x N

        # transforming the components to ensure they're in the right ranges
        parsed['read_strengths'] = 1 + tf.nn.softplus(parsed['read_strengths'])
        parsed['write_strength'] = 1 + tf.nn.softplus(parsed['write_strength'])
        parsed['erase_vector'] = tf.nn.sigmoid(parsed['erase_vector'])
        parsed['free_gates'] = tf.nn.sigmoid(parsed['free_gates'])
        parsed['allocation_gate'] = tf.nn.sigmoid(parsed['allocation_gate'])
        parsed['write_gate'] = tf.nn.sigmoid(parsed['write_gate'])
        parsed['read_modes'] = tf.nn.softmax(parsed['read_modes'], 1)
        parsed['cache_gate'] = tf.nn.sigmoid(parsed['cache_gate'])

        return parsed # dict of tensors

