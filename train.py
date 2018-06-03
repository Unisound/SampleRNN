from __future__ import print_function
import argparse
from datetime import datetime
import json
import os
import sys
import time

import librosa
import numpy as np
import tensorflow as tf

from tensorflow.python.client import timeline
from samplernn import SampleRnnModel
from samplernn import AudioReader
from samplernn import mu_law_decode
from samplernn import optimizer_factory

DATA_DIRECTORY = './pinao-corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 5
GENERATE_EVERY = 10
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-3
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = None
MOMENTUM = 0.9
MAX_TO_KEEP = 5

N_SECS = 3
SAMPLE_RATE = 22050
LENGTH = N_SECS * SAMPLE_RATE

BATCH_SIZE = 1
NUM_GPU = 1


def get_arguments():
    parser = argparse.ArgumentParser(description='SampleRnn example network')
    parser.add_argument('--num_gpus',         type=int,   default=NUM_GPU)
    parser.add_argument('--batch_size',       type=int,   default=BATCH_SIZE)
    parser.add_argument('--data_dir',         type=str,
                        default=DATA_DIRECTORY)
    parser.add_argument('--logdir_root',      type=str,   default=LOGDIR_ROOT)
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY)
    parser.add_argument('--num_steps',        type=int,   default=NUM_STEPS)
    parser.add_argument('--learning_rate',    type=float,
                        default=LEARNING_RATE)
    parser.add_argument('--sample_size',      type=int,   default=SAMPLE_SIZE)
    parser.add_argument('--sample_rate',      type=int,   default=SAMPLE_RATE)
    parser.add_argument('--l2_regularization_strength',
                        type=float, default=L2_REGULARIZATION_STRENGTH)
    parser.add_argument('--silence_threshold', type=float,
                        default=SILENCE_THRESHOLD)
    parser.add_argument('--optimizer',        type=str,
                        default='adam', choices=optimizer_factory.keys())
    parser.add_argument('--momentum',         type=float, default=MOMENTUM)

    parser.add_argument('--seq_len',          type=int, required=True)
    parser.add_argument('--big_frame_size',   type=int, required=True)
    parser.add_argument('--frame_size',       type=int, required=True)
    parser.add_argument('--q_levels',         type=int, required=True)
    parser.add_argument('--dim',              type=int, required=True)
    parser.add_argument('--n_rnn',            type=int,
                        choices=list(range(1, 6)), required=True)
    parser.add_argument('--emb_size',         type=int, required=True)
    parser.add_argument('--rnn_type', choices=['LSTM', 'GRU'], required=True)
    parser.add_argument('--max_checkpoints',  type=int, default=MAX_TO_KEEP)
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def create_model(args):
    # Create network.
    net = SampleRnnModel(
        batch_size=args.batch_size,
        big_frame_size=args.big_frame_size,
        frame_size=args.frame_size,
        q_levels=args.q_levels,
        rnn_type=args.rnn_type,
        dim=args.dim,
        n_rnn=args.n_rnn,
        seq_len=args.seq_len,
        emb_size=args.emb_size)
    return net


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

# GENERATE


def create_gen_wav_para(net):
    with tf.name_scope('infe_para'):
        infe_para = dict()
        infe_para['infe_big_frame_inp'] = tf.get_variable(
            "infe_big_frame_inp",
            [
                net.batch_size,
                net.big_frame_size,
                1
            ],
            dtype=tf.float32
        )
        infe_para['infe_big_frame_outp'] = tf.get_variable(
            "infe_big_frame_outp",
            [
                net.batch_size,
                net.big_frame_size / net.frame_size,
                net.dim
            ],
            dtype=tf.float32)

        infe_para['infe_big_frame_outp_slices'] = tf.get_variable(
            "infe_big_frame_outp_slices",
            [
                net.batch_size,
                1,
                net.dim
            ],
            dtype=tf.float32
        )
        infe_para['infe_frame_inp'] = tf.get_variable(
            "infe_frame_inp",
            [
                net.batch_size,
                net.frame_size,
                1
            ],
            dtype=tf.float32
        )
        infe_para['infe_frame_outp'] = tf.get_variable(
            "infe_frame_outp",
            [
                net.batch_size,
                net.frame_size,
                net.dim
            ],
            dtype=tf.float32
        )

        infe_para['infe_frame_outp_slices'] = tf.get_variable(
            "infe_frame_outp_slices",
            [
                net.batch_size,
                1,
                net.dim
            ],
            dtype=tf.float32
        )
        infe_para['infe_sample_inp'] = tf.get_variable(
            "infe_sample_inp",
            [
                net.batch_size,
                net.frame_size,
                1
            ],
            dtype=tf.int32
        )

        infe_para['infe_big_frame_state'] = net.big_cell.zero_state(
            net.batch_size,
            tf.float32
        )
        infe_para['infe_frame_state'] = net.cell.zero_state(
            net.batch_size,
            tf.float32
        )

        tf.get_variable_scope().reuse_variables()
        infe_para['infe_big_frame_outp'], \
            infe_para[
                'infe_final_big_frame_state'
        ] = net._create_network_BigFrame(
                num_steps=1,
                big_frame_state=infe_para['infe_big_frame_state'],
                big_input_sequences=infe_para['infe_big_frame_inp']
        )

        infe_para['infe_frame_outp'], \
            infe_para['infe_final_frame_state'] = net._create_network_Frame(
                num_steps=1,
                big_frame_outputs=infe_para['infe_big_frame_outp_slices'],
                frame_state=infe_para['infe_frame_state'],
                input_sequences=infe_para['infe_frame_inp']
        )

        sample_out = net._create_network_Sample(
            frame_outputs=infe_para['infe_frame_outp_slices'],
            sample_input_sequences=infe_para['infe_sample_inp']
        )
        sample_out = tf.reshape(
            sample_out,
            [-1, net.q_levels]
        )
        infe_para['infe_sample_outp'] = tf.cast(
            tf.nn.softmax(
                tf.cast(
                    sample_out,
                    tf.float64
                )
            ),
            tf.float32
        )

        infe_para['infe_sample_decode_inp'] = tf.placeholder(
            tf.int32
        )
        infe_para['infe_decode'] = mu_law_decode(
            infe_para['infe_sample_decode_inp'],
            net.q_levels
        )

        return infe_para


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def generate_and_save_samples(step, net, infe_para, sess):
    samples = np.zeros((net.batch_size, LENGTH, 1), dtype='int32')
    samples[:, :net.big_frame_size, :] = np.int32(net.q_levels//2)

    final_big_s, final_s = sess.run([net.big_initial_state, net.initial_state])
    big_frame_out = None
    frame_out = None
    sample_out = None
    for t in range(net.big_frame_size, LENGTH):
        # big frame
        if t % net.big_frame_size == 0:
            big_frame_out = None
            big_input_sequences = samples[
                :,
                t - net.big_frame_size:t,
                :
            ].astype('float32')
            big_frame_out, final_big_s = sess.run(
                [
                    infe_para[
                        'infe_big_frame_outp'
                    ],
                    infe_para['infe_final_big_frame_state']
                ],
                feed_dict={
                    infe_para['infe_big_frame_inp']: big_input_sequences,
                    infe_para['infe_big_frame_state']: final_big_s
                }
            )
        # frame
        if t % net.frame_size == 0:
            frame_input_sequences = samples[
                :,
                t - net.frame_size:t,
                :
            ].astype('float32')
            big_frame_output_idx = (
                t // net.frame_size
            ) % (
                net.big_frame_size // net.frame_size
            )
            frame_out, final_s = sess.run(
                [
                    infe_para['infe_frame_outp'],
                    infe_para['infe_final_frame_state']
                ],
                feed_dict={
                    infe_para['infe_big_frame_outp_slices']: big_frame_out[
                        :,
                        [big_frame_output_idx],
                        :
                    ],
                    infe_para['infe_frame_inp']: frame_input_sequences,
                    infe_para['infe_frame_state']: final_s
                }
            )
        # sample
        sample_input_sequences = samples[:, t-net.frame_size:t, :]
        frame_output_idx = t % net.frame_size
        sample_out = sess.run(
            infe_para['infe_sample_outp'],
            feed_dict={
                infe_para['infe_frame_outp_slices']: frame_out[
                    :,
                    [frame_output_idx],
                    :
                ],
                infe_para['infe_sample_inp']: sample_input_sequences
            }
        )
        sample_next_list = []
        for row in sample_out:
            sample_next = np.random.choice(
                np.arange(net.q_levels), p=row)
            sample_next_list.append(sample_next)
        samples[:, t] = np.array(sample_next_list).reshape([-1, 1])
    for i in range(net.batch_size):
        inp = samples[i].reshape([-1, 1]).tolist()
        out = sess.run(infe_para['infe_decode'],
                       feed_dict={infe_para['infe_sample_decode_inp']: inp})
        write_wav(out, SAMPLE_RATE, './generated/test_' +
                  str(step)+'_'+str(i)+'.wav')
        if i >= 10:
            break


def main():
    args = get_arguments()
    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    logdir = os.path.join(args.logdir_root, 'train')
    coord = tf.train.Coordinator()

    with tf.name_scope('create_inputs'):
        reader = AudioReader(args.data_dir,
                             coord,
                             sample_rate=args.sample_rate,
                             sample_size=args.sample_size,
                             silence_threshold=args.silence_threshold)
        audio_batch = reader.dequeue(args.batch_size)
    net = create_model(args)
    global_step = tf.get_variable(
        'global_step',
        [],
        initializer=tf.constant_initializer(0),
        trainable=False
    )
    optim = optimizer_factory[args.optimizer](
        learning_rate=args.learning_rate,
        momentum=args.momentum)
######Multi GPU###########
    tower_grads = []
    losses = []
    train_input_batch_rnn = []
    train_big_frame_state = []
    train_frame_state = []
    final_big_frame_state = []
    final_frame_state = []
    for i in range(args.num_gpus):
        train_input_batch_rnn.append(
            tf.Variable(
                tf.zeros(
                    [net.batch_size, net.seq_len, 1]
                ),
                trainable=False,
                name="input_batch_rnn",
                dtype=tf.float32
            )
        )
        train_big_frame_state.append(
            net.big_cell.zero_state(net.batch_size, tf.float32))
        final_big_frame_state.append(
            net.big_cell.zero_state(net.batch_size, tf.float32))
        train_frame_state.append(
            net.cell.zero_state(net.batch_size, tf.float32))
        final_frame_state.append(
            net.cell.zero_state(net.batch_size, tf.float32))
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(args.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('TOWER_%d' % (i)) as scope:
                    # Create model.
                    print("Creating model On Gpu:%d." % (i))
                    (
                        loss,
                        final_big_frame_state[i],
                        final_frame_state[i]=net.loss_SampleRnn(
                            train_input_batch_rnn[i],
                            train_big_frame_state[i],
                            train_frame_state[i],
                            l2_regularization_strength=args.l2_regularization_strength  # noqa: E501
                        )
                    )
                    tf.get_variable_scope().reuse_variables()
                    losses.append(loss)
                    # Reuse variables for the next tower.
                    trainable = tf.trainable_variables()
                    gradients = optim.compute_gradients(
                        loss,
                        trainable,
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N   # noqa: E501
                    )
                    tower_grads.append(gradients)
    grad_vars = average_gradients(tower_grads)
    grads, vars = list(zip(*grad_vars))
    grads_clipped, _ = tf.clip_by_global_norm(grads, 5.0)
    grad_vars = list(zip(grads_clipped, vars))

    for name in grad_vars:
        print(name)

    apply_gradient_op = optim.apply_gradients(
        grad_vars, global_step=global_step)
#################
    infe_para = create_gen_wav_para(net)

    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    #run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    tf_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(var_list=tf.trainable_variables(),
                           max_to_keep=args.max_checkpoints)

    try:
        saved_global_step = load(saver, sess, logdir)
        if saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    step = None
    last_saved_step = saved_global_step
    try:
        for step in range(saved_global_step + 1, args.num_steps):
            if (step-1) % GENERATE_EVERY == 0 and step > GENERATE_EVERY:
                generate_and_save_samples(step, net, infe_para, sess)

            final_big_s = []
            final_s = []
            for g in range(args.num_gpus):
                final_big_s.append(sess.run(net.big_initial_state))
                final_s.append(sess.run(net.initial_state))
            start_time = time.time()
            inputslist = [sess.run(audio_batch) for i in range(args.num_gpus)]
            loss_sum = 0
            idx_begin = 0
            audio_length = args.sample_size - args.big_frame_size
            bptt_length = args.seq_len - args.big_frame_size
            stateful_rnn_length = audio_length // bptt_length
            outp_list = [summaries,
                         losses,
                         apply_gradient_op,
                         final_big_frame_state,
                         final_frame_state]
            for i in range(0, stateful_rnn_length):
                inp_dict = {}
                for g in range(args.num_gpus):
                    inp_dict[train_input_batch_rnn[g]] = \
                        inputslist[g][:, idx_begin: idx_begin+args.seq_len, :]
                    inp_dict[train_big_frame_state[g]] = final_big_s[g]
                    inp_dict[train_frame_state[g]] = final_s[g]
                idx_begin += args.seq_len-args.big_frame_size

                summary, loss_gpus, _, final_big_s, final_s = \
                    sess.run(outp_list, feed_dict=inp_dict)
                writer.add_summary(summary, step)
                for g in range(args.num_gpus):
                    loss_gpu = loss_gpus[g]/stateful_rnn_length
                    loss_sum += loss_gpu/args.num_gpus
            duration = time.time() - start_time
            print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'
                  .format(step, loss_sum, duration))

            if step % args.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
