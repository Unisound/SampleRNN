import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from .ops import mu_law_encode


class SampleRnnModel(object):
    def __init__(self, batch_size, big_frame_size, frame_size,
                 q_levels, rnn_type, dim, n_rnn, seq_len, emb_size):
        self.batch_size = batch_size
        self.big_frame_size = big_frame_size
        self.frame_size = frame_size
        self.q_levels = q_levels
        self.rnn_type = rnn_type
        self.dim = dim
        self.n_rnn = n_rnn
        self.seq_len = seq_len
        self.emb_size = emb_size

        def single_cell():
            return tf.contrib.rnn.GRUCell(self.dim)
        if 'LSTM' == self.rnn_type:
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(self.dim)
        self.cell = single_cell()
        self.big_cell = single_cell()
        if self.n_rnn > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell(
                [single_cell() for _ in range(self.n_rnn)])
            self.big_cell = tf.contrib.rnn.MultiRNNCell(
                [single_cell() for _ in range(self.n_rnn)])
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
        self.big_initial_state = self.big_cell.zero_state(
            self.batch_size, tf.float32)

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=self.q_levels,
                dtype=tf.float32)
            shape = [self.batch_size, -1, self.q_levels]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def _create_network_BigFrame(self,
                                 num_steps,
                                 big_frame_state,
                                 big_input_sequences):
        with tf.variable_scope('BigFrame_layer'):
            big_input_frames = tf.reshape(big_input_sequences, [
                tf.shape(big_input_sequences)[0],
                tf.shape(big_input_sequences)[1] // self.big_frame_size,
                self.big_frame_size])
            big_input_frames = (big_input_frames / self.q_levels/2.0) - 1.0
            big_input_frames *= 2.0

            big_frame_outputs = []
            big_frame_proj_weights = tf.get_variable(
                "big_frame_proj_weights",
                [
                    self.dim,
                    self.dim * self.big_frame_size / self.frame_size
                ],
                dtype=tf.float32
            )
            with tf.variable_scope("BIG_FRAME_RNN"):
                for time_step in range(num_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (big_frame_cell_output, big_frame_state) = self.big_cell(
                        big_input_frames[:, time_step, :], big_frame_state)
                    big_frame_outputs.append(math_ops.matmul(
                        big_frame_cell_output, big_frame_proj_weights))
                final_big_frame_state = big_frame_state
            big_frame_outputs = tf.stack(big_frame_outputs)
            big_frame_outputs = tf.transpose(big_frame_outputs, perm=[1, 0, 2])
            big_frame_outputs = tf.reshape(
                big_frame_outputs,
                [
                    tf.shape(big_frame_outputs)[0],
                    tf.shape(big_frame_outputs)[
                        1] * self.big_frame_size // self.frame_size,
                    -1,
                ]
            )
            return big_frame_outputs, final_big_frame_state

    def _create_network_Frame(self,
                              num_steps,
                              big_frame_outputs,
                              frame_state,
                              input_sequences):
        with tf.variable_scope('Frame_layer'):
            input_frames = tf.reshape(input_sequences, [
                tf.shape(input_sequences)[0],
                tf.shape(input_sequences)[1] // self.frame_size,
                self.frame_size])
            input_frames = (input_frames / self.q_levels/2.0) - 1.0
            input_frames *= 2.0

            frame_outputs = []
            frame_proj_weights = tf.get_variable(
                "frame_proj_weights",
                [
                    self.dim,
                    self.dim * self.frame_size,
                ],
                dtype=tf.float32,
            )
            frame_cell_proj_weights = tf.get_variable(
                "frame_cell_proj_weights",
                [
                    self.frame_size,
                    self.dim,
                ],
                dtype=tf.float32,
            )
            with tf.variable_scope("FRAME_RNN"):
                for time_step in range(num_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    cell_input = tf.reshape(
                        input_frames[:, time_step, :], [-1, self.frame_size])
                    cell_input = math_ops.matmul(
                        cell_input, frame_cell_proj_weights)
                    cell_input = cell_input + \
                        tf.reshape(
                            big_frame_outputs[:, time_step, :], [-1, self.dim])
                    (frame_cell_output, frame_state) = self.cell(
                        cell_input, frame_state)

                    frame_outputs.append(math_ops.matmul(
                        frame_cell_output, frame_proj_weights))
            final_frame_state = frame_state
            frame_outputs = tf.stack(frame_outputs)
            frame_outputs = tf.transpose(frame_outputs, perm=[1, 0, 2])

            frame_outputs = tf.reshape(frame_outputs,
                                       [tf.shape(frame_outputs)[0],
                                        tf.shape(frame_outputs)[
                                           1] * self.frame_size,
                                        -1])
            return frame_outputs, final_frame_state

    def _create_network_Sample(self,
                               frame_outputs,
                               sample_input_sequences):
        with tf.variable_scope('Sample_layer'):
            sample_shap = [tf.shape(sample_input_sequences)[0],
                           tf.shape(sample_input_sequences)[1]*self.emb_size,
                           1]
            embedding = tf.get_variable(
                "embedding", [self.q_levels, self.emb_size])
            sample_input_sequences = embedding_ops.embedding_lookup(
                embedding, tf.reshape(sample_input_sequences, [-1]))
            sample_input_sequences = tf.reshape(
                sample_input_sequences, sample_shap)

            '''Create a convolution filter variable with the specified name and shape,
      and initialize it using Xavier initialition.'''
            filter_initializer = tf.contrib.layers.xavier_initializer_conv2d()
            sample_filter_shape = [self.emb_size*2, 1, self.dim]
            sample_filter = tf.get_variable(
                "sample_filter",
                sample_filter_shape,
                initializer=filter_initializer
            )
            out = tf.nn.conv1d(sample_input_sequences,
                               sample_filter,
                               stride=self.emb_size,
                               padding="VALID",
                               name="sample_conv")
            out = out + frame_outputs
            sample_mlp1_weights = tf.get_variable(
                "sample_mlp1", [self.dim, self.dim], dtype=tf.float32)
            sample_mlp2_weights = tf.get_variable(
                "sample_mlp2", [self.dim, self.dim], dtype=tf.float32)
            sample_mlp3_weights = tf.get_variable(
                "sample_mlp3", [self.dim, self.q_levels], dtype=tf.float32)
            out = tf.reshape(out, [-1, self.dim])
            out = math_ops.matmul(out, sample_mlp1_weights)
            out = tf.nn.relu(out)
            out = math_ops.matmul(out, sample_mlp2_weights)
            out = tf.nn.relu(out)
            out = math_ops.matmul(out, sample_mlp3_weights)
            out = tf.reshape(out, [-1, sample_shap[1] //
                                   self.emb_size - 1, self.q_levels])
            return out

    def _create_network_SampleRnn(self,
                                  train_big_frame_state,
                                  train_frame_state):
        with tf.name_scope('SampleRnn_net'):
            # big frame
            big_input_sequences = tf.cast(self.encoded_input_rnn, tf.float32)[
                :, :-self.big_frame_size, :]
            big_frame_num_steps = (
                self.seq_len-self.big_frame_size) // self.big_frame_size
            big_frame_outputs,\
                final_big_frame_state = self._create_network_BigFrame(
                    num_steps=big_frame_num_steps,
                    big_frame_state=train_big_frame_state,
                    big_input_sequences=big_input_sequences,
                )
            # frame
            input_sequences = tf.cast(self.encoded_input_rnn, tf.float32)[
                :,
                self.big_frame_size - self.frame_size: -self.frame_size,
                :
            ]
            frame_num_steps = (
                self.seq_len-self.big_frame_size) // self.frame_size
            frame_outputs, final_frame_state = \
                self._create_network_Frame(num_steps=frame_num_steps,
                                           big_frame_outputs=big_frame_outputs,
                                           frame_state=train_frame_state,
                                           input_sequences=input_sequences)
            # sample
            sample_input_sequences = self.encoded_input_rnn[
                :,
                self.big_frame_size - self.frame_size: -1,
                :
            ]
            sample_output = self._create_network_Sample(
                frame_outputs,
                sample_input_sequences=sample_input_sequences,
            )
            return sample_output, final_big_frame_state, final_frame_state

    def loss_SampleRnn(self,
                       train_input_batch_rnn,
                       train_big_frame_state,
                       train_frame_state,
                       l2_regularization_strength=None,
                       name='sample'):
        with tf.name_scope(name):
            self.encoded_input_rnn = mu_law_encode(
                train_input_batch_rnn, self.q_levels)
            encoded_rnn = self._one_hot(self.encoded_input_rnn)
            raw_output, final_big_frame_state, final_frame_state = \
                self._create_network_SampleRnn(
                    train_big_frame_state, train_frame_state)
            with tf.name_scope('loss'):
                target_output_rnn = encoded_rnn[:, self.big_frame_size:, :]
                target_output_rnn = tf.reshape(
                    target_output_rnn, [-1, self.q_levels])
                prediction = tf.reshape(raw_output, [-1, self.q_levels])

                loss = tf.nn.softmax_cross_entropy_with_logits(
                    logits=prediction,
                    labels=target_output_rnn)
                reduced_loss = tf.reduce_mean(loss)
                tf.summary.scalar('loss', reduced_loss)
                if l2_regularization_strength is None:
                    return (
                        reduced_loss,
                        final_big_frame_state,
                        final_frame_state
                    )
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if not('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (reduced_loss +
                                  l2_regularization_strength * l2_loss)

                    tf.summary.scalar('l2_loss', l2_loss)
                    tf.summary.scalar('total_loss', total_loss)

                    return total_loss, final_big_frame_state, final_frame_state
