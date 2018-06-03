import fnmatch
import os
import random
import re
import threading

import librosa
import sys
import copy
import numpy as np
import tensorflow as tf


def randomize_files(files):
    files_idx = [i for i in range(len(files))]
    random.shuffle(files_idx)

    for idx in range(len(files)):
        yield files[files_idx[idx]]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename


def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self, audio_dir, coord, sample_rate, sample_size=None,
                 silence_threshold=None, queue_size=32):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(
            queue_size, ['float32'], shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        audio_list = []
        iterator = load_generic_audio(self.audio_dir, self.sample_rate)
        for audio, _ in iterator:
            audio_list.append(audio)
        while not stop:
            for audio_copy in audio_list:
                audio = copy.deepcopy(audio_copy)
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    audio = trim_silence(audio[:, 0], self.silence_threshold)
                    audio = audio.reshape(-1, 1)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio.")
                pad_elements = self.sample_size - 1 - \
                    (audio.shape[0] + self.sample_size - 1) % self.sample_size
                audio = np.concatenate(
                    [
                        audio,
                        np.full(
                            (pad_elements, 1),
                            0.0,
                            dtype='float32'
                        )
                    ],
                    axis=0
                )
                if self.sample_size:
                    while len(audio) >= self.sample_size:
                        piece = audio[:self.sample_size, :]
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        audio = audio[self.sample_size:, :]
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: audio})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
