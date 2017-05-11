from __future__ import division
import tensorflow as tf

def create_adam_optimizer(learning_rate, momentum):
  return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)

def create_sgd_optimizer(learning_rate, momentum):
  return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)

def create_rmsprop_optimizer(learning_rate, momentum):
  return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)

optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}

def mu_law_encode(audio, quantization_channels):
  '''Quantizes waveform amplitudes.'''
  with tf.name_scope('encode'):
    mu = quantization_channels - 1
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
    magnitude = tf.log(1 + mu * safe_audio_abs) / tf.log(1. + mu)
    signal = tf.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return tf.cast((signal + 1) / 2 * mu + 0.5, tf.int32)

def mu_law_decode(output, quantization_channels):
  '''Recovers waveform from quantized values.'''
  with tf.name_scope('decode'):
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    casted = tf.cast(output, tf.float32)
    signal = 2 * (casted / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    return tf.sign(signal) * magnitude
