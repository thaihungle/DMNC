import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from memory import Memory
import utility
import os

class DNC:

    def __init__(self, controller_class, input_size, output_size, max_sequence_length=100,
                 memory_words_num = 256, memory_word_size = 64, memory_read_heads = 4,
                 batch_size = 1,hidden_controller_dim=256, use_emb=True,
                 use_mem=True, decoder_mode=False, emb_size=64,
                 write_protect=False, dual_controller=False, dual_emb=True,
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
        self.input_size = input_size
        self.output_size = output_size
        self.max_sequence_length = max_sequence_length
        self.words_num = memory_words_num
        self.word_size = memory_word_size
        self.read_heads = memory_read_heads
        self.batch_size = batch_size
        self.unpacked_input_data = None
        self.packed_output = None
        self.packed_memory_view = None
        self.decoder_mode = decoder_mode
        self.decoder_point = tf.placeholder(tf.int32, name='decoder_point')#
        self.emb_size = emb_size
        self.emb_size2 = emb_size
        self.dual_emb = dual_emb
        self.use_mem = use_mem
        self.use_emb = use_emb
        self.hidden_controller_dim = hidden_controller_dim
        self.attend_dim = attend_dim
        self.use_teacher = use_teacher
        self.teacher_force = tf.placeholder(tf.bool,[None], name='teacher')
        self.persist_mode = persist_mode
        self.clear_mem = tf.placeholder(tf.bool,None, name='clear_mem')

        if self.use_emb is False:
            self.emb_size=input_size

        if self.use_emb is False:
            self.emb_size2=output_size

        if self.attend_dim>0:
            self.W_a = tf.get_variable('W_a', [hidden_controller_dim, self.attend_dim],
                                       initializer=tf.random_normal_initializer(stddev=0.1))
            self.U_a = tf.get_variable('U_a', [hidden_controller_dim, self.attend_dim],
                                       initializer=tf.random_normal_initializer(stddev=0.1))

            self.v_a = tf.get_variable('v_a', [self.attend_dim],
                                       initializer=tf.random_normal_initializer(stddev=0.1))

        # DNC (or NTM) should be structurized into 2 main modules:
        # all the graph is setup inside these twos:
        self.W_emb_encoder = tf.get_variable('embe_w', [self.input_size, self.emb_size],
                                             initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
        self.W_emb_decoder = tf.get_variable('embd_w', [self.output_size, self.emb_size],
                                             initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

        self.memory = Memory(self.words_num, self.word_size, self.read_heads, self.batch_size)
        self.controller = controller_class(self.emb_size, self.output_size, self.read_heads,
                                           self.word_size, self.batch_size, use_mem, hidden_dim=hidden_controller_dim)
        self.dual_controller = dual_controller
        if self.dual_controller:
            with tf.variable_scope('controller2_scope'):
                if attend_dim==0:
                    self.controller2 = controller_class(self.emb_size2, self.output_size, self.read_heads,
                                                       self.word_size, self.batch_size, use_mem, hidden_dim=hidden_controller_dim)
                else:
                    self.controller2 = controller_class(self.emb_size2+hidden_controller_dim, self.output_size, self.read_heads,
                                                        self.word_size, self.batch_size, use_mem,
                                                        hidden_dim=hidden_controller_dim)
        self.write_protect = write_protect

        # input data placeholders
        self.input_data = tf.placeholder(tf.float32, [batch_size, None, input_size], name='input')
        self.target_output = tf.placeholder(tf.float32, [batch_size, None, output_size], name='targets')
        self.mask = tf.placeholder(tf.bool, [batch_size, None], name='mask')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')# variant length?

        if persist_mode:
            self.cur_c = tf.get_variable('cur_c', [self.batch_size, hidden_controller_dim],
                                                   trainable=False)
            self.assign_op_cur_c = self.cur_c.assign(np.ones([self.batch_size, hidden_controller_dim]) * 1e-6)
            self.cur_h = tf.get_variable('cur_h', [self.batch_size, hidden_controller_dim],
                                         trainable=False)
            self.assign_op_cur_h = self.cur_h.assign(np.ones([self.batch_size, hidden_controller_dim]) * 1e-6)
            self.cur_mem_content = tf.get_variable('cur_mc', [self.batch_size, self.words_num, self.word_size],trainable=False)
            self.assign_op_cur_mem = self.cur_mem_content.assign(np.ones([self.batch_size, self.words_num, self.word_size])*1e-6)
            self.cur_u = tf.get_variable('cur_u', [self.batch_size, self.words_num],trainable=False) # initial usage vector u
            self.assign_op_cur_u = self.cur_u.assign(np.zeros([self.batch_size, self.words_num]))
            self.cur_p = tf.get_variable('cur_p',[self.batch_size, self.words_num], trainable=False) # initial precedence vector p
            self.assign_op_cur_p = self.cur_p.assign(np.zeros([self.batch_size, self.words_num]))
            self.cur_L = tf.get_variable('cur_L',[self.batch_size, self.words_num, self.words_num], trainable=False) # initial link matrix L
            self.assign_op_cur_L = self.cur_L.assign(np.ones([self.batch_size, self.words_num, self.words_num])*1e-6)
            self.cur_ww = tf.get_variable('cur_ww',[self.batch_size, self.words_num], trainable=False) # initial write weighting
            self.assign_op_cur_ww = self.cur_ww.assign(np.ones([self.batch_size, self.words_num])*1e-6)
            self.cur_rw = tf.get_variable('cur_rw',[self.batch_size, self.words_num, self.read_heads], trainable=False) # initial read weightings
            self.assign_op_cur_rw = self.cur_rw.assign(np.ones([self.batch_size, self.words_num, self.read_heads])*1e-6)
            self.cur_rv = tf.get_variable('cur_rv',[self.batch_size, self.word_size, self.read_heads], trainable=False)  # initial read vectors
            self.assign_op_cur_rv = self.cur_rv.assign(np.ones([self.batch_size, self.word_size, self.read_heads])*1e-6)


        self.build_graph()


    # The nature of DNC is to process data by step and remmeber data at each time step when necessary
    # If input has sequence format --> suitable with RNN core controller --> each time step in RNN equals 1 time step in DNC
    # or just feed input to MLP --> each feed is 1 time step
    def _step_op(self, time, step, memory_state, controller_state=None, controller_hiddens=None):
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

        last_read_vectors = memory_state[6] # read values from memory
        pre_output, interface, nn_state = None, None, None

        # compute outputs from controller
        if self.controller.has_recurrent_nn:
            # controller state is the rnn cell state pass through each time step
            def c1():
                if not self.use_emb:
                    step2 = tf.reshape(step, [-1, self.input_size])
                    return self.controller.process_input(step2, last_read_vectors, controller_state)
                else:
                    return self.controller.process_input(step, last_read_vectors, controller_state)
            def c2():
                if not self.use_emb:
                    step2=tf.reshape(step,[-1,self.output_size])
                else:
                    step2=step
                #attention

                if controller_hiddens:
                    values = utility.pack_into_tensor2(controller_hiddens, axis=1)[:,:self.decoder_point]
                    # values=controller_hiddens.gather(tf.range(0,self.decoder_point))
                    encoder_outputs =\
                        tf.reshape(values,[self.batch_size,-1,self.hidden_controller_dim])  # bs x Lin x h
                    v = tf.tanh(
                        tf.reshape(tf.matmul(tf.reshape(encoder_outputs, [-1, self.hidden_controller_dim]), self.U_a),
                                   [self.batch_size, -1, self.attend_dim])
                        + tf.reshape(
                            tf.matmul(tf.reshape(controller_state[1], [-1, self.hidden_controller_dim]), self.W_a),
                            [self.batch_size, 1, self.attend_dim]))  # bs.Lin x h_att
                    v = tf.reshape(v, [-1, self.attend_dim])
                    eijs = tf.matmul(v, tf.expand_dims(self.v_a,1)) # bs.Lin x 1
                    eijs = tf.reshape(eijs,[self.batch_size,-1])# bs x Lin
                    exps = tf.exp(eijs)
                    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1]) # bs x Lin
                    att = tf.reduce_sum(encoder_outputs*tf.expand_dims(alphas,2), 1) # bs x h x 1
                    att = tf.reshape(att,[self.batch_size, self.hidden_controller_dim]) # bs x h
                    step2=tf.concat([step2,att], axis=-1) #bs x (decoder_is + h)



                return self.controller2.process_input(step2, last_read_vectors, controller_state)

            if self.dual_controller:
                pre_output, interface, nn_state = tf.cond(time>=self.decoder_point, c2, c1)
            else:
                pre_output, interface, nn_state = self.controller.process_input(step, last_read_vectors, controller_state)
        else:
            pre_output, interface = self.controller.process_input(step, last_read_vectors)

        # memory_matrix isthe copy of memory for reading process later
        # do the write first

        def fn1():
            return self.memory.write(
            memory_state[0], memory_state[1], memory_state[5],
            memory_state[4], memory_state[2], memory_state[3],
            interface['write_key'],
            interface['write_strength'],
            interface['free_gates'],
            interface['allocation_gate'],
            interface['write_gate'],
            interface['write_vector'],
            interface['erase_vector'],
        )

        def fn2():
            return memory_state[1], memory_state[4], memory_state[0], memory_state[3], memory_state[2]



        if self.write_protect:
            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector\
                = tf.cond(time>=self.decoder_point, fn2, fn1)
        else:
            usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.memory.write(
                memory_state[0], memory_state[1], memory_state[5],
                memory_state[4], memory_state[2], memory_state[3],
                interface['write_key'],
                interface['write_strength'],
                interface['free_gates'],
                interface['allocation_gate'],
                interface['write_gate'],
                interface['write_vector'],
                interface['erase_vector']
            )

        # then do the read, read after write because the write weight is needed to produce temporal linklage to guide the reading
        read_weightings, read_vectors = self.memory.read(
            memory_matrix,
            memory_state[5],
            interface['read_keys'],
            interface['read_strengths'],
            link_matrix,
            interface['read_modes'],
        )

        return [
            # report new memory state to be updated outside the condition branch
            memory_matrix, #0

            # neccesary for next step to compute memory stuffs
            usage_vector, #1
            precedence_vector, #2
            link_matrix, #3
            write_weighting, #4
            read_weightings, #5
            read_vectors, #6

            # the final output of dnc
            self.controller.final_output(pre_output, read_vectors), #7

            # the values public info to outside
            interface['read_modes'], #8
            interface['allocation_gate'], #9
            interface['write_gate'], #10

            # report new state of RNN if exists, neccesary for next step to compute inner controller stuff
            nn_state[0] if nn_state is not None else tf.zeros(1), #11
            nn_state[1] if nn_state is not None else tf.zeros(1) #12
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
                return tf.matmul(self.unpacked_input_data.read(time), self.W_emb_encoder)
        def fn2():
            def fn2_1():
                return self.target_output[:,time-1,:]
            def fn2_2():
                return tf.one_hot(tf.argmax(outputs_cache.read(time - 1), axis=-1), depth=self.output_size)
            if self.use_teacher:
                feed_value=tf.cond(self.teacher_force[time-1],fn2_1,fn2_2)
            else:
                feed_value=fn2_2()


            if not self.use_emb:
                #return outputs_cache.read(time - 1)
                r = feed_value
                r = tf.reshape(r,[self.batch_size,self.output_size])
                print(r.shape)
                return r
            elif self.dual_emb:
                return tf.matmul(feed_value, self.W_emb_decoder)
            else:
                return tf.matmul(feed_value, self.W_emb_encoder)
            # if self.dual_emb:
            #     return tf.matmul(tf.nn.softmax(outputs_cache.read(time-1),dim=-1), self.W_emb_decoder)
            # else:
            #     return tf.matmul(tf.nn.softmax(outputs_cache.read(time - 1),dim=-1), self.W_emb_encoder)


        if self.decoder_mode:
            step_input = tf.cond(time>=self.decoder_point, fn2, fn1)
        else:
            if self.use_emb:
                step_input = tf.matmul(self.unpacked_input_data.read(time), self.W_emb_encoder)
            else:
                step_input = self.unpacked_input_data.read(time)


        # compute one step of controller
        if self.attend_dim>0:
            output_list = self._step_op(time, step_input, memory_state, controller_state, controller_hiddens)
        else:
            output_list = self._step_op(time, step_input, memory_state, controller_state)
        # update memory parameters

        # new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:7])
        new_controller_state = LSTMStateTuple(output_list[11], output_list[12]) #  state hidden values

        controller_hiddens = controller_hiddens.write(time, output_list[12])
        outputs = outputs.write(time, output_list[7])# new output is updated
        outputs_cache = outputs_cache.write(time, output_list[7])# new output is updated
        # collecting memory view for the current step
        free_gates = free_gates.write(time, output_list[8])
        allocation_gates = allocation_gates.write(time, output_list[9])
        write_gates = write_gates.write(time, output_list[10])
        read_weightings = read_weightings.write(time, output_list[5])
        write_weightings = write_weightings.write(time, output_list[4])
        usage_vectors = usage_vectors.write(time, output_list[1])

        # all variables have been updated should be return for next step reference
        return (
            time + 1, #0
            new_memory_state, #1
            outputs, #2
            free_gates,allocation_gates, write_gates, #3 4 5
            read_weightings, write_weightings, usage_vectors, #6 7 8
            new_controller_state, #9
            outputs_cache,  #10
            controller_hiddens, #11
        )


    def build_graph(self):
        """
        builds the computational graph that performs a step-by-step evaluation
        of the input data batches
        """

        # make dynamic time step length tensor
        self.unpacked_input_data = utility.unpack_into_tensorarray(self.input_data, 1, self.sequence_length)

        # want to store all time step values of these variables
        outputs = tf.TensorArray(tf.float32, self.sequence_length)
        outputs_cache = tf.TensorArray(tf.float32, self.sequence_length)
        free_gates = tf.TensorArray(tf.float32, self.sequence_length)
        allocation_gates = tf.TensorArray(tf.float32, self.sequence_length)
        write_gates = tf.TensorArray(tf.float32, self.sequence_length)
        read_weightings = tf.TensorArray(tf.float32, self.sequence_length)
        write_weightings = tf.TensorArray(tf.float32, self.sequence_length)
        usage_vectors = tf.TensorArray(tf.float32, self.sequence_length)
        controller_hiddens = tf.TensorArray(tf.float32, self.sequence_length, clear_after_read=False)



        # inital state for RNN controller
        controller_state = self.controller.get_state() if self.controller.has_recurrent_nn else (tf.zeros(1), tf.zeros(1))
        print(controller_state)
        memory_state = self.memory.init_memory()
        if self.persist_mode:
            def p1():
                return memory_state, controller_state
            def p2():
                return (self.cur_mem_content, self.cur_u, self.cur_p,
                        self.cur_L, self.cur_ww, self.cur_rw, self.cur_rv), LSTMStateTuple(self.cur_c, self.cur_h)
            memory_state, controller_state=tf.cond(self.clear_mem, p1, p2)
        if not isinstance(controller_state, LSTMStateTuple):
            try:
                controller_state = LSTMStateTuple(controller_state[0], controller_state[1])
                print('seq state hid')
            except:
                print('dddd')

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
                    usage_vectors, controller_state,
                    outputs_cache, controller_hiddens
                ), # do not need to provide intial values, the initial value lies in the variables themselves
                parallel_iterations=1,
                swap_memory=True
            )
        self.cur_mem_content, self.cur_u, self.cur_p, \
        self.cur_L, self.cur_ww, self.cur_rw, self.cur_rv = final_results[1]
        self.cur_c = final_results[9][0]
        self.cur_h = final_results[9][1]
        dependencies = []
        if self.controller.has_recurrent_nn:
            # tensor array of pair of hidden and state values of rnn
            dependencies.append(self.controller.update_state(final_results[9]))

        with tf.control_dependencies(dependencies):
            # convert output tensor array to normal tensor
            self.packed_output = utility.pack_into_tensor(final_results[2], axis=1)
            self.packed_memory_view = {
                'free_gates': utility.pack_into_tensor(final_results[3], axis=1),
                'allocation_gates': utility.pack_into_tensor(final_results[4], axis=1),
                'write_gates': utility.pack_into_tensor(final_results[5], axis=1),
                'read_weightings': utility.pack_into_tensor(final_results[6], axis=1),
                'write_weightings': utility.pack_into_tensor(final_results[7], axis=1),
                'usage_vectors': utility.pack_into_tensor(final_results[8], axis=1),
                'final_controller_ch':final_results[9],
            }


    def get_outputs(self):
        """
        returns the graph nodes for the output and memory view
        Returns: Tuple
            outputs: Tensor (batch_size, time_steps, output_size)
            memory_view: dict
        """
        return self.packed_output, self.packed_memory_view

    def assign_pretrain_emb_encoder(self, sess, lookup_mat):
        assign_op_W_emb_encoder = self.W_emb_encoder.assign(lookup_mat)
        sess.run([assign_op_W_emb_encoder])

    def assign_pretrain_emb_decoder(self, sess, lookup_mat):
        assign_op_W_emb_decoder = self.W_emb_decoder.assign(lookup_mat)
        sess.run([assign_op_W_emb_decoder])

    def build_loss_function_multiple(self, optimizer=None, output_sizes=[]):
        print('build loss....')
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer()
        output, _ = self.get_outputs()

        target = tf.slice(self.target_output, [0, 0, 0],
                 [self.batch_size, self.sequence_length, output_sizes[0]])
        subout = tf.slice(output, [0, 0, 0],
                          [self.batch_size, self.sequence_length, output_sizes[0]])

        prob = tf.nn.softmax(subout, dim=-1)
        probs = [prob]
        subouts=[subout]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.slice(target, [0, self.decoder_point, 0],
                            [self.batch_size, self.sequence_length - self.decoder_point, output_sizes[0]]),
            logits=tf.slice(prob, [0, self.decoder_point, 0],
                            [self.batch_size, self.sequence_length - self.decoder_point, output_sizes[0]]), dim=-1)
        )
        for ii,si in enumerate(output_sizes[1:]):
            target = tf.slice(self.target_output, [0, 0, output_sizes[ii]],
                              [self.batch_size, self.sequence_length, si])
            subout = tf.slice(output, [0, 0, output_sizes[ii]],
                              [self.batch_size, self.sequence_length, si])

            prob = tf.nn.softmax(subout, dim=-1)
            probs += [prob]
            subouts+=[subout]
            loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.slice(target, [0, self.decoder_point, 0],
                                [self.batch_size, self.sequence_length - self.decoder_point, si]),
                logits=tf.slice(prob, [0, self.decoder_point, 0],
                                [self.batch_size, self.sequence_length - self.decoder_point, si]), dim=-1)
            )


        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_value(grad, -10, 10), var)


        apply_gradients = optimizer.apply_gradients(gradients)
        return subouts, probs, loss, apply_gradients

    def build_loss_function(self, optimizer=None, clip_s=10):
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
        return '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(self.use_mem,
                                       self.decoder_mode,
                                       self.dual_controller,
                                       self.write_protect,
                                       self.words_num,
                                       self.word_size,
                                       self.use_teacher,
                                       self.attend_dim,
                                       self.persist_mode)


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

    def clear_current_mem(self,sess):
        if self.persist_mode:
            sess.run([self.assign_op_cur_mem, self.assign_op_cur_u, self.assign_op_cur_p,
                      self.assign_op_cur_L, self.assign_op_cur_ww, self.assign_op_cur_rw, self.assign_op_cur_rv])

            sess.run([self.assign_op_cur_c, self.assign_op_cur_h])

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
    def get_bool_rand_incremental(size_seq, prob_true_min=0, prob_true_max=0.25):
        ret = []
        for i in range(size_seq):
            prob_true=(prob_true_max-prob_true_min)/size_seq*i
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)

    @staticmethod
    def get_bool_rand_curriculum(size_seq, epoch, k=0.99, type='exp'):
        if type=='exp':
            prob_true = k**epoch
        elif type=='sig':
            prob_true = k / (k + np.exp(epoch / k))
        ret = []
        for i in range(size_seq):
            if np.random.rand() < prob_true:
                ret.append(True)
            else:
                ret.append(False)
        return np.asarray(ret)