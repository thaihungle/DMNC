import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from memory import Memory
import utility
import os
import numpy as np

class Dual_DNC:

    def __init__(self, controller_class, input_size1, input_size2, output_size,
                 memory_words_num = 256, memory_word_size = 64, memory_read_heads = 4,
                 batch_size = 1, hidden_controller_dim=128,
                 use_mem=True, decoder_mode=False, emb_size=64,
                 write_protect=False, dual_emb=True, share_mem=False,
                 use_teacher=False, attend_dim=0, persist_mode=False):
        """
        constructs a complete DNC architecture as described in the DNC paper
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html
        Parameters:
        -----------
        controller_class: BaseController
            a concrete implementation of the BaseController class
        input_size: int
            the size of the input vector
        output_size: int
            the size of the output vector
        max_sequence_length: int
            the maximum length of an input sequence
        memory_words_num: int
            the number of words that can be stored in memory
        memory_word_size: int
            the size of an individual word in memory
        memory_read_heads: int
            the number of read heads in the memory
        batch_size: int
            the size of the data batch
        """
        saved_args = locals()
        print("saved_args is", saved_args)
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.output_size = output_size
        self.words_num = memory_words_num
        self.word_size = memory_word_size
        self.read_heads = memory_read_heads
        self.batch_size = batch_size
        self.unpacked_input_data1 = None
        self.unpacked_input_data2 = None
        self.packed_output = None
        self.packed_memory_view = None
        self.decoder_mode = decoder_mode
        self.decoder_point = tf.placeholder(tf.int32, name='decoder_point')#
        self.encode1_point = tf.placeholder(tf.int32, name='encode1_point')#
        self.encode2_point = tf.placeholder(tf.int32, name='encode2_point')
        self.emb_size = emb_size
        self.use_mem=use_mem
        self.share_mem=share_mem
        self.use_teacher = use_teacher
        self.attend_dim = attend_dim
        self.hidden_controller_dim = hidden_controller_dim
        self.teacher_force = tf.placeholder(tf.bool,[None], name='teacher')
        self.persist_mode = persist_mode
        self.clear_mem = tf.placeholder(tf.bool, None, name='clear_mem')

        if self.attend_dim>0:
            self.W_a1 = tf.get_variable('W_a1', [hidden_controller_dim, self.attend_dim],
                                      initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.U_a1 = tf.get_variable('U_a1', [hidden_controller_dim, self.attend_dim],
                                  initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

            self.v_a1 = tf.get_variable('v_a1', [self.attend_dim],
                                  initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

            self.W_a2 = tf.get_variable('W_a2', [hidden_controller_dim, self.attend_dim],
                                        initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.U_a2 = tf.get_variable('U_a2', [hidden_controller_dim, self.attend_dim],
                                        initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

            self.v_a2 = tf.get_variable('v_a2', [self.attend_dim],
                                    initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

        # DNC (or NTM) should be structurized into 2 main modules:
        # all the graph is setup inside these twos:
        self.W_emb1_encoder = tf.get_variable('embe1_w', [self.input_size1, self.emb_size],
                                             initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        self.W_emb2_encoder = tf.get_variable('embe2_w', [self.input_size2, self.emb_size],
                                             initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        self.W_emb_decoder = tf.get_variable('embd_w', [self.output_size, self.emb_size],
                                             initializer=tf.random_uniform_initializer(minval=-1, maxval=1))


        with tf.variable_scope('input1_scope'):
            self.memory1 = Memory(self.words_num, self.word_size, self.read_heads, self.batch_size)
            self.controller1 = controller_class(self.emb_size, self.output_size, self.read_heads,
                                            self.word_size, self.batch_size, use_mem, hidden_dim=hidden_controller_dim)

        with tf.variable_scope('input2_scope'):
            if not share_mem:
                self.memory2 = Memory(self.words_num, self.word_size, self.read_heads, self.batch_size)
            else:
                self.memory2=self.memory1
            self.controller2 = controller_class(self.emb_size, self.output_size, self.read_heads,
                                               self.word_size, self.batch_size, use_mem, hidden_dim=hidden_controller_dim)

        with tf.variable_scope('output_scope'):
            if self.attend_dim==0:
                self.controller3 = controller_class(self.emb_size, self.output_size, self.read_heads,
                                                   self.word_size, self.batch_size, use_mem, is_two_mem=2,
                                                    hidden_dim=hidden_controller_dim*2)
            else:
                self.controller3 = controller_class(self.emb_size+hidden_controller_dim * 2, self.output_size, self.read_heads,
                                                    self.word_size, self.batch_size, use_mem, is_two_mem=2,
                                                    hidden_dim=hidden_controller_dim * 2)

        self.write_protect = write_protect

        # input data placeholders
        self.input_data1 = tf.placeholder(tf.float32, [batch_size, None, input_size1], name='input')
        self.input_data2 = tf.placeholder(tf.float32, [batch_size, None, input_size2], name='input')
        self.target_output = tf.placeholder(tf.float32, [batch_size, None, output_size], name='targets')
        self.mask = tf.placeholder(tf.bool, [batch_size, None], name='mask')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')# variant length?
        self.dual_emb = dual_emb


        if persist_mode:
            self.cur_c = []
            self.assign_op_cur_c = []
            self.cur_h = []
            self.assign_op_cur_h = []

            self.cur_mem_content = []
            self.assign_op_cur_mem = []
            self.cur_u = []
            self.assign_op_cur_u = []
            self.cur_p = []
            self.assign_op_cur_p = []
            self.cur_L = []
            self.assign_op_cur_L = []
            self.cur_ww = []
            self.assign_op_cur_ww = []
            self.cur_rw = []
            self.assign_op_cur_rw = []
            self.cur_rv = []
            self.assign_op_cur_rv = []

            for i in range(2):
                self.cur_c += [tf.get_variable('cur_c{}'.format(i), [self.batch_size, hidden_controller_dim],
                                             trainable=False)]
                self.assign_op_cur_c += [self.cur_c[i].assign(np.ones([self.batch_size, hidden_controller_dim]) * 1e-6)]
                self.cur_h += [tf.get_variable('cur_h{}'.format(i), [self.batch_size, hidden_controller_dim],
                                             trainable=False)]
                self.assign_op_cur_h += [self.cur_h[i].assign(np.ones([self.batch_size, hidden_controller_dim]) * 1e-6)]

                self.cur_mem_content+=[tf.get_variable('cur_mc{}'.format(i), [self.batch_size, self.words_num, self.word_size],
                                                       trainable=False)]
                self.assign_op_cur_mem+=[self.cur_mem_content[i].assign(
                    np.ones([self.batch_size, self.words_num, self.word_size]) * 1e-6)]
                self.cur_u += [tf.get_variable('cur_u{}'.format(i), [self.batch_size, self.words_num],
                                             trainable=False)]  # initial usage vector u
                self.assign_op_cur_u += [self.cur_u[i].assign(np.zeros([self.batch_size, self.words_num]))]
                self.cur_p += [tf.get_variable('cur_p{}'.format(i), [self.batch_size, self.words_num],
                                             trainable=False)] # initial precedence vector p
                self.assign_op_cur_p += [self.cur_p[i].assign(np.zeros([self.batch_size, self.words_num]))]
                self.cur_L += [tf.get_variable('cur_L{}'.format(i), [self.batch_size, self.words_num, self.words_num],
                                             trainable=False)]  # initial link matrix L
                self.assign_op_cur_L += [self.cur_L[i].assign(np.ones([self.batch_size, self.words_num, self.words_num]) * 1e-6)]
                self.cur_ww += [tf.get_variable('cur_ww{}'.format(i), [self.batch_size, self.words_num],
                                              trainable=False)]  # initial write weighting
                self.assign_op_cur_ww += [self.cur_ww[i].assign(np.ones([self.batch_size, self.words_num]) * 1e-6)]
                self.cur_rw += [tf.get_variable('cur_rw{}'.format(i), [self.batch_size, self.words_num, self.read_heads],
                                              trainable=False)]  # initial read weightings
                self.assign_op_cur_rw += [self.cur_rw[i].assign(np.ones([self.batch_size, self.words_num, self.read_heads]) * 1e-6)]
                self.cur_rv += [tf.get_variable('cur_rv{}'.format(i), [self.batch_size, self.word_size, self.read_heads],
                                              trainable=False)]  # initial read vectors
                self.assign_op_cur_rv += [self.cur_rv[i].assign(np.ones([self.batch_size, self.word_size, self.read_heads]) * 1e-6)]

        self.build_graph()


    # The nature of DNC is to process data by step and remmeber data at each time step when necessary
    # If input has sequence format --> suitable with RNN core controller --> each time step in RNN equals 1 time step in DNC
    # or just feed input to MLP --> each feed is 1 time step
    def _step_op(self, time, step1, step2, memory_state, controller_state=None, controller_hiddens=None):
        """
        performs a step operation on the input step data
        Parameters:
        ----------
        step: Tensor (batch_size, input_size)
        memory_state: Tuple
            a tuple of current memory parameters
        controller_state: Tuple
            the state of the controller if it's recurrent
        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        memory_state1 = memory_state[0]
        memory_state2 = memory_state[1]
        last_read_vectors1 = memory_state1[6]  # read values from memory
        last_read_vectors2 = memory_state2[6]  # read values from memory

        controller_state1 = controller_state[0]
        controller_state2 = controller_state[1]

        # controller state is the rnn cell state pass through each time step
        def c1():

            def c11():
                return self.controller1.process_zero()

            def c12():
                return self.controller1.process_input(step1, last_read_vectors1, controller_state1)


            pre_output1, interface1, nn_state1 = tf.cond(time<self.encode1_point, c11, c12)

            def c13():
                return  self.controller2.process_zero()

            def c14():
                return  self.controller2.process_input(step2, last_read_vectors2, controller_state2)

            pre_output2, interface2, nn_state2 = tf.cond(time<self.encode2_point, c13, c14)

            pre_output12 = pre_output1 + pre_output2
            interface12 = (interface1, interface2)
            nn_state12 = (nn_state1, nn_state2)

            return pre_output12, interface12, nn_state12

        def c2():
            con_c1=controller_state1[0]
            con_h1=controller_state1[1]
            con_c2 = controller_state2[0]
            con_h2 = controller_state2[1]

            ncontroller_state = LSTMStateTuple(tf.concat([con_c1,con_c2],axis=-1), tf.concat([con_h1,con_h2],axis=-1))
            nread_vec = tf.concat([last_read_vectors1, last_read_vectors2],axis=1)

            step = step1
            if controller_hiddens:
                from_steps=[self.encode1_point, self.encode2_point]
                v_a=[self.v_a1, self.v_a2]
                U_a=[self.U_a1, self.U_a2]
                W_a=[self.W_a1, self.W_a2]
                for cci, controller_hiddens_ in enumerate(controller_hiddens):
                    values = controller_hiddens_.gather(tf.range(from_steps[cci], self.decoder_point))
                    encoder_outputs = \
                        tf.reshape(values, [self.batch_size, -1, self.hidden_controller_dim])  # bs x Lin x h


                    v = tf.tanh(
                        tf.reshape(tf.matmul(tf.reshape(encoder_outputs, [-1, self.hidden_controller_dim]), U_a[cci]),
                                   [self.batch_size, -1, self.attend_dim])
                        + tf.reshape(
                            tf.matmul(tf.reshape(controller_state[cci][0], [-1, self.hidden_controller_dim]), W_a[cci]),
                            [self.batch_size, 1, self.attend_dim]))  # bs.Lin x h_att
                    v = tf.reshape(v, [-1, self.attend_dim])
                    eijs = tf.matmul(v, tf.expand_dims(v_a[cci], 1))  # bs.Lin x 1
                    eijs = tf.reshape(eijs, [self.batch_size, -1])  # bs x Lin
                    exps = tf.exp(eijs)
                    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])  # bs x Lin
                    att = tf.reduce_sum(encoder_outputs * tf.expand_dims(alphas, 2), 1)  # bs x h x 1
                    att = tf.reshape(att, [self.batch_size, self.hidden_controller_dim])  # bs x h
                    step = tf.concat([step, att], axis=-1)  # bs x (decoder_is + h)

            pre_output, interface, nn_state = \
                self.controller3.process_input(step,
                                               nread_vec,
                                               ncontroller_state)
            #trick split than group
            c_l, c_r = tf.split(nn_state[0],num_or_size_splits=2, axis=-1)
            h_l, h_r = tf.split(nn_state[1], num_or_size_splits=2, axis=-1)
            return pre_output, interface, (LSTMStateTuple(c_l,h_l), LSTMStateTuple(c_r, h_r))


        pre_output, interface, nn_state = tf.cond(time>=self.decoder_point, c2, c1)

        interface1 = interface[0]
        interface2 = interface[1]


        # memory_matrix isthe copy of memory for reading process later
        # do the write first

        def fn1():

            def fn11():
                return memory_state1[1], memory_state1[4], memory_state1[0], memory_state1[3], memory_state1[2]

            def fn12():
                return self.memory1.write(
                    memory_state1[0], memory_state1[1], memory_state1[5],
                    memory_state1[4], memory_state1[2], memory_state1[3],
                    interface1['write_key'],
                    interface1['write_strength'],
                    interface1['free_gates'],
                    interface1['allocation_gate'],
                    interface1['write_gate'],
                    interface1['write_vector'],
                    interface1['erase_vector']
                )

            def fn13():
                return memory_state2[1], memory_state2[4], memory_state2[0], memory_state2[3], memory_state2[2]

            def fn14():
                return self.memory2.write(
                memory_state2[0], memory_state2[1], memory_state2[5],
                memory_state2[4], memory_state2[2], memory_state2[3],
                interface2['write_key'],
                interface2['write_strength'],
                interface2['free_gates'],
                interface2['allocation_gate'],
                interface2['write_gate'],
                interface2['write_vector'],
                interface2['erase_vector'])

            usage_vector1, write_weighting1, memory_matrix1, link_matrix1, precedence_vector1 = \
                tf.cond(time<self.encode1_point, fn11, fn12)
            usage_vector2, write_weighting2, memory_matrix2, link_matrix2, precedence_vector2 = \
                tf.cond(time<self.encode2_point, fn13, fn14)

            usage_vector12 = (usage_vector1, usage_vector2)
            write_weighting12 = (write_weighting1, write_weighting2)
            memory_matrix12 = (memory_matrix1, memory_matrix2)
            link_matrix12 = (link_matrix1, link_matrix2)
            precedence_vector12 = (precedence_vector1, precedence_vector2)

            return  usage_vector12, write_weighting12, memory_matrix12, link_matrix12, precedence_vector12


        def fn2():
            return (memory_state1[1],memory_state2[1]), \
                   (memory_state1[4], memory_state2[4]), \
                   (memory_state1[0], memory_state2[0]), \
                   (memory_state1[3], memory_state2[3]), \
                   (memory_state1[2], memory_state2[2])



        if self.write_protect:
            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector\
                = tf.cond(time>=self.decoder_point, fn2, fn1)
        else:
            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = fn1()


        # then do the read, read after write because the write weight is needed to produce temporal linklage to guide the reading

        def r11():
            return self.memory1.read_zero()

        def r12():
            return self.memory1.read(
            memory_matrix[0],
            memory_state1[5],
            interface1['read_keys'],
            interface1['read_strengths'],
            link_matrix[0],
            interface1['read_modes'],
        )

        def r13():
            return self.memory2.read_zero()

        def r14():
            return self.memory2.read(
            memory_matrix[1],
            memory_state2[5],
            interface2['read_keys'],
            interface2['read_strengths'],
            link_matrix[1],
            interface2['read_modes'],
        )

        read_weightings1, read_vectors1 = tf.cond(time<self.encode1_point, r11, r12)

        read_weightings2, read_vectors2 = tf.cond(time<self.encode2_point, r13, r14)

        return [
            # report new memory state to be updated outside the condition branch
            memory_matrix, #0

            # neccesary for next step to compute memory stuffs
            usage_vector, #1
            precedence_vector, #2
            link_matrix, #3
            write_weighting, #4
            (read_weightings1, read_weightings2), #5
            (read_vectors1, read_vectors2), #6

            # the final output of dnc
            self.controller3.final_output(pre_output, tf.concat([read_vectors1, read_vectors2], axis=1)), #7

            # the values public info to outside
            (interface1['free_gates'], interface2['free_gates']), #8
            (interface1['allocation_gate'], interface2['allocation_gate']), #9
            (interface1['write_gate'],interface2['write_gate']), #10

            # report new state of RNN if exists, neccesary for next step to compute inner controller stuff
            nn_state[0][0] if nn_state[0] is not None else tf.zeros(1), #11
            nn_state[0][1] if nn_state[0] is not None else tf.zeros(1), #12
            nn_state[1][0] if nn_state[1] is not None else tf.zeros(1) , # 13
            nn_state[1][1] if nn_state[1] is not None else tf.zeros(1)  # 14
        ]

    '''
    THIS WRAPPER FOR ONE STEP OF COMPUTATION --> INTERFACE FOR SCAN/WHILE LOOP
    '''
    def _loop_body(self, time, memory_state, outputs, free_gates, allocation_gates, write_gates,
                   read_weightings, write_weightings, usage_vectors, controller_state,
                   outputs_cache, controller_hiddens):
        """
        the body of the DNC sequence processing loop
        Parameters:
        ----------
        time: Tensor
        outputs: TensorArray
        memory_state: Tuple
        free_gates: TensorArray
        allocation_gates: TensorArray
        write_gates: TensorArray
        read_weightings: TensorArray,
        write_weightings: TensorArray,
        usage_vectors: TensorArray,
        controller_state: Tuple
        Returns: Tuple containing all updated arguments
        """

        # dynamic tensor array input

        def fn1():
            return tf.matmul(self.unpacked_input_data1.read(time), self.W_emb1_encoder)
        def fn2():
            def fn2_1():
                return self.target_output[:,time-1,:]
            def fn2_2():
                return tf.one_hot(tf.argmax(outputs_cache.read(time - 1), axis=-1), depth=self.output_size)
            if self.use_teacher:
                feed_value=tf.cond(self.teacher_force[time-1],fn2_1,fn2_2)
            else:
                feed_value=fn2_2()

            if self.dual_emb:
                return tf.matmul(feed_value, self.W_emb_decoder)
            else:
                return tf.matmul(feed_value, self.W_emb1_encoder)

        def fn12():
            return tf.matmul(self.unpacked_input_data2.read(time), self.W_emb2_encoder)
        def fn22():
            return tf.zeros([self.batch_size, self.emb_size]) #here for format consistent, not used


        if self.decoder_mode:
            step_input1 = tf.cond(time>=self.decoder_point, fn2, fn1)
            step_input2 = tf.cond(time >= self.decoder_point, fn22, fn12)
        else:
            step_input1 = fn1()
            step_input2 = fn12()

        # compute one step of controller
        if self.attend_dim>0:
            output_list = self._step_op(time, step_input1, step_input2, memory_state, controller_state, controller_hiddens)
        else:
            output_list = self._step_op(time, step_input1, step_input2, memory_state, controller_state)

        # update memory parameters

        new_memory_state1=[]
        new_memory_state2=[]
        for obj in output_list[:7]:
            new_memory_state1.append(obj[0])
            new_memory_state2.append(obj[1])

        new_memory_state = [tuple(new_memory_state1), tuple(new_memory_state2)]
        new_controller_state = [LSTMStateTuple(output_list[11], output_list[12]),
                                LSTMStateTuple(output_list[13], output_list[14])] # hidden and state values

        controller_hiddens = [controller_hiddens[0].write(time, output_list[11]),
                              controller_hiddens[1].write(time, output_list[13])]

        outputs = outputs.write(time, output_list[7])# new output is updated
        outputs_cache = outputs_cache.write(time, output_list[7])# new output is updated
        # collecting memory view for the current step
        free_gates2 = [free_gates[0].write(time, output_list[8][0]),free_gates[1].write(time, output_list[8][1])]
        allocation_gates2 = [allocation_gates[0].write(time, output_list[9][0]),allocation_gates[1].write(time, output_list[9][1])]
        write_gates2 = [write_gates[0].write(time, output_list[10][0]),write_gates[1].write(time, output_list[10][1])]
        read_weightings2 = [read_weightings[0].write(time, output_list[5][0]),read_weightings[1].write(time, output_list[5][1])]
        write_weightings2 =[write_weightings[0].write(time, output_list[4][0]),write_weightings[1].write(time, output_list[4][1])]
        usage_vectors2 = [usage_vectors[0].write(time, output_list[1][0]),usage_vectors[1].write(time, output_list[1][1])]

        # all variables have been updated should be return for next step reference
        return (
            time + 1, #0
            new_memory_state, #1
            outputs, #2
            free_gates2,allocation_gates2, write_gates2, #3 4 5
            read_weightings2, write_weightings2, usage_vectors2, #6 7 8
            new_controller_state, #9
            outputs_cache,  #10
            controller_hiddens #11
        )


    def build_graph(self):
        """
        builds the computational graph that performs a step-by-step evaluation
        of the input data batches
        """

        # make dynamic time step length tensor
        self.unpacked_input_data1 = utility.unpack_into_tensorarray(self.input_data1, 1, self.sequence_length)
        self.unpacked_input_data2 = utility.unpack_into_tensorarray(self.input_data2, 1, self.sequence_length)

        # want to store all time step values of these variables
        outputs = tf.TensorArray(tf.float32, self.sequence_length)
        outputs_cache = tf.TensorArray(tf.float32, self.sequence_length)
        free_gates = [tf.TensorArray(tf.float32, self.sequence_length),tf.TensorArray(tf.float32, self.sequence_length)]
        allocation_gates = [tf.TensorArray(tf.float32, self.sequence_length), tf.TensorArray(tf.float32, self.sequence_length)]
        write_gates = [tf.TensorArray(tf.float32, self.sequence_length),tf.TensorArray(tf.float32, self.sequence_length)]
        read_weightings = [tf.TensorArray(tf.float32, self.sequence_length),tf.TensorArray(tf.float32, self.sequence_length)]
        write_weightings = [tf.TensorArray(tf.float32, self.sequence_length),tf.TensorArray(tf.float32, self.sequence_length)]
        usage_vectors = [tf.TensorArray(tf.float32, self.sequence_length),tf.TensorArray(tf.float32, self.sequence_length)]
        controller_hiddens = [tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False),
                              tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)]


        # inital state for RNN controller
        controller_state1 = self.controller1.get_state() if self.controller1.has_recurrent_nn else (tf.zeros(1), tf.zeros(1))
        controller_state2 = self.controller2.get_state() if self.controller2.has_recurrent_nn else (tf.zeros(1), tf.zeros(1))
        memory_state = [self.memory1.init_memory(), self.memory2.init_memory()]

        if self.persist_mode:
            def p1():
                return memory_state, controller_state1, controller_state2
            def p2():
                tmp=[(self.cur_mem_content[0], self.cur_u[0], self.cur_p[0],
                  self.cur_L[0], self.cur_ww[0], self.cur_rw[0], self.cur_rv[0]),
                 (self.cur_mem_content[1], self.cur_u[1], self.cur_p[1],
                  self.cur_L[1], self.cur_ww[1], self.cur_rw[1], self.cur_rv[1])
                 ]
                if len(memory_state[0])>len(tmp[0]):
                    print('cache mode')
                    tmp[0] = (self.cur_mem_content[0], self.cur_u[0], self.cur_p[0],
                              self.cur_L[0], self.cur_ww[0], self.cur_rw[0], self.cur_rv[0],
                              memory_state[0][-2],memory_state[0][-1])
                    tmp[1] = (self.cur_mem_content[1], self.cur_u[1], self.cur_p[1],
                              self.cur_L[1], self.cur_ww[1], self.cur_rw[1], self.cur_rv[1],
                              memory_state[1][-2], memory_state[1][-1])
                return tmp, \
                       LSTMStateTuple(self.cur_c[0], self.cur_h[0]),LSTMStateTuple(self.cur_c[1], self.cur_h[1])
            memory_state, controller_state1, controller_state2=tf.cond(self.clear_mem, p1, p2)

        if not isinstance(controller_state1, LSTMStateTuple):
            controller_state1 = LSTMStateTuple(controller_state1[0], controller_state1[1])
        if not isinstance(controller_state2, LSTMStateTuple):
            controller_state2 = LSTMStateTuple(controller_state2[0], controller_state2[1])

        controller_state=[controller_state1, controller_state2]



        # final_results = None

        with tf.variable_scope("sequence_loop"):
            time = tf.constant(0, dtype=tf.int32)

            # use while instead of scan --> suitable with dynamic time step
            final_results = tf.while_loop(
                cond=lambda time, *_: time < self.sequence_length,
                body=self._loop_body,
                loop_vars=(
                    time, memory_state, outputs,
                    free_gates, allocation_gates, write_gates,
                    read_weightings, write_weightings,
                    usage_vectors, controller_state, outputs_cache,controller_hiddens
                ), # do not need to provide intial values, the initial value lies in the variables themselves
                parallel_iterations=1,
                swap_memory=True,
            )

        dependencies = []
        if self.controller1.has_recurrent_nn:
            # tensor array of pair of hidden and state values of rnn
            dependencies.append(self.controller1.update_state(final_results[9][0]))
        if self.controller2.has_recurrent_nn:
            # tensor array of pair of hidden and state values of rnn
            dependencies.append(self.controller2.update_state(final_results[9][1]))

        with tf.control_dependencies(dependencies):
            # convert output tensor array to normal tensor
            self.packed_output = utility.pack_into_tensor(final_results[2], axis=1)
            self.packed_memory_view = {
                'free_gates1': utility.pack_into_tensor(final_results[3][0], axis=1),
                'free_gates2': utility.pack_into_tensor(final_results[3][1], axis=1),
                'allocation_gates1': utility.pack_into_tensor(final_results[4][0], axis=1),
                'allocation_gates2': utility.pack_into_tensor(final_results[4][1], axis=1),
                'write_gates1': utility.pack_into_tensor(final_results[5][0], axis=1),
                'write_gates2': utility.pack_into_tensor(final_results[5][1], axis=1),
                'read_weightings1': utility.pack_into_tensor(final_results[6][0], axis=1),
                'read_weightings2': utility.pack_into_tensor(final_results[6][1], axis=1),
                'write_weightings1': utility.pack_into_tensor(final_results[7][0], axis=1),
                'write_weightings2': utility.pack_into_tensor(final_results[7][1], axis=1),
                'usage_vectors1': utility.pack_into_tensor(final_results[8][0], axis=1),
                'usage_vectors2': utility.pack_into_tensor(final_results[8][1], axis=1),
            }


    def get_outputs(self):
        """
        returns the graph nodes for the output and memory view
        Returns: Tuple
            outputs: Tensor (batch_size, time_steps, output_size)
            memory_view: dict
        """
        return self.packed_output, self.packed_memory_view

    def assign_pretrain_emb1_encoder(self, sess, lookup_mat):
        assign_op_W_emb_encoder = self.W_emb1_encoder.assign(lookup_mat)
        sess.run([assign_op_W_emb_encoder])

    def assign_pretrain_emb2_encoder(self, sess, lookup_mat):
        assign_op_W_emb_encoder = self.W_emb2_encoder.assign(lookup_mat)
        sess.run([assign_op_W_emb_encoder])

    def assign_pretrain_emb_decoder(self, sess, lookup_mat):
        assign_op_W_emb_decoder = self.W_emb_decoder.assign(lookup_mat)
        sess.run([assign_op_W_emb_decoder])

    def build_loss_function(self, optimizer=None,clip_s=10):
        print('build loss....')
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()
        output, _ = self.get_outputs()
        prob = tf.nn.softmax(output, dim=-1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.slice(self.target_output, [0, self.decoder_point, 0],
                            [self.batch_size, self.sequence_length - self.decoder_point, self.output_size]),
            logits=tf.slice(output, [0, self.decoder_point, 0],
                            [self.batch_size, self.sequence_length - self.decoder_point, self.output_size]), dim=-1)
        )


        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_value(grad, -clip_s, clip_s), var)


        apply_gradients = optimizer.apply_gradients(gradients)
        return output, prob, loss, apply_gradients

    def build_loss_function_multi_label(self, optimizer=None, clip_s=10):
        print('build loss....')
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()
        output, _ = self.get_outputs()
        prob = tf.nn.sigmoid(output)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.slice(self.target_output, [0, self.decoder_point, 0],
                            [self.batch_size, self.sequence_length - self.decoder_point, self.output_size]),
            logits=tf.slice(output, [0, self.decoder_point, 0],
                            [self.batch_size, self.sequence_length - self.decoder_point, self.output_size]))
        )

        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_value(grad, -clip_s, clip_s), var)

        apply_gradients = optimizer.apply_gradients(gradients)
        return output, prob, loss, apply_gradients


    def build_loss_function_mask(self, optimizer=None, clip_s=10):
        print('build loss mask....')
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()
        output, _ = self.get_outputs()
        prob = tf.nn.softmax(output, dim=-1)

        score=tf.nn.softmax_cross_entropy_with_logits(
            labels=self.target_output,
            logits=output, dim=-1)
        score_flatten=tf.reshape(score,[-1])
        mask_flatten=tf.reshape(self.mask,[-1])
        mask_score=tf.boolean_mask(score_flatten, mask_flatten)


        loss = tf.reduce_mean(mask_score)


        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_value(grad, -clip_s, clip_s), var)


        apply_gradients = optimizer.apply_gradients(gradients)
        return output, prob, loss, apply_gradients



    def print_config(self):
        return 'din_sout{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(self.use_mem,
                                       self.decoder_mode,
                                       self.write_protect,
                                       self.words_num,
                                       self.word_size,
                                       self.share_mem,
                                       self.use_teacher,
                                       self.persist_mode,
                                       self.attend_dim)


    @staticmethod
    def save(session, ckpts_dir, name):
        """
        saves the current values of the model's parameters to a checkpoint
        Parameters:
        ----------
        session: tf.Session
            the tensorflow session to save
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        checkpoint_dir = os.path.join(ckpts_dir, name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        tf.train.Saver(tf.trainable_variables()).save(session, os.path.join(checkpoint_dir, 'model.ckpt'))


    @staticmethod
    def restore(session, ckpts_dir, name):
        """
        session: tf.Session
            the tensorflow session to restore into
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        tf.train.Saver(tf.trainable_variables()).restore(session, os.path.join(ckpts_dir, name, 'model.ckpt'))


    def clear_current_mem(self,sess):
        if self.persist_mode:
            for i in range(2):
                sess.run([self.assign_op_cur_mem[i], self.assign_op_cur_u[i], self.assign_op_cur_p[i],
                          self.assign_op_cur_L[i], self.assign_op_cur_ww[i], self.assign_op_cur_rw[i],
                          self.assign_op_cur_rv[i]])

                sess.run([self.assign_op_cur_c[i], self.assign_op_cur_h[i]])

    @staticmethod
    def get_bool_rand_incremental(size_seq, prob_true_min=0, prob_true_max=0.25):
        ret = []
        for i in range(size_seq):
            prob_true = (prob_true_max - prob_true_min) / size_seq * i
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)

    @staticmethod
    def get_bool_rand(size_seq, prob_true=0.1):
        ret = []
        for i in range(size_seq):
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)

    @staticmethod
    def get_bool_rand_curriculum(size_seq, epoch, k=0.99, type='exp'):
        if type == 'exp':
            prob_true = k ** epoch
        elif type == 'sig':
            prob_true = k / (k + np.exp(epoch / k))
        ret = []
        for i in range(size_seq):
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)