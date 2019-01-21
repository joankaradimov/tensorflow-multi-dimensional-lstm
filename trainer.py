from time import time

import argparse
import logging
import numpy as np
import tensorflow.contrib.slim as slim

from data_random_short_diagonal import next_batch, visualise_mat, get_relevant_prediction_index
from md_lstm import *

logger = logging.getLogger(__name__)


class FileLogger(object):
    def __init__(self, full_filename, headers):
        self._headers = headers
        self._out_fp = open(full_filename, 'w')
        self._write(headers)

    def write(self, line):
        assert len(line) == len(self._headers)
        self._write(line)

    def close(self):
        self._out_fp.close()

    def _write(self, arr):
        arr = [str(e) for e in arr]
        self._out_fp.write(' '.join(arr) + '\n')
        self._out_fp.flush()


def run():
    learning_rate = 0.01
    batch_size = 16
    h = 8
    w = 8
    channels = 1
    hidden_size = 16

    x = tf.placeholder(tf.float32, [batch_size, h, w, channels])
    y = tf.placeholder(tf.float32, [batch_size, h, w, channels])

    rnn_out, _ = MdRnnWhileLoop()(rnn_size=hidden_size, input_data=x)

    model_out = slim.fully_connected(inputs=rnn_out,
                                     num_outputs=1,
                                     activation_fn=tf.nn.sigmoid)

    loss = tf.reduce_mean(tf.square(y - model_out))
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    fp = FileLogger('out.tsv', ['steps', 'overall_loss', 'time'])
    steps = 1000
    for i in range(steps):
        batch = next_batch(batch_size, h, w)
        grad_step_start_time = time()
        batch_x = np.expand_dims(batch[0], axis=3)
        batch_y = np.expand_dims(batch[1], axis=3)

        model_preds, tot_loss_value, _ = sess.run([model_out, loss, grad_update], feed_dict={
            x: batch_x,
            y: batch_y,
        })

        """
        ____________
        |          |
        |          |
        |     x    |
        |      x <----- extract this prediction. Relevant loss is only computed for this value.
        |__________|    we don't care about the rest (even though the model is trained on all values
                        for simplicity). A standard LSTM should have a very high value for relevant loss
                        whereas a MD LSTM (which can see all the TOP LEFT corner) should perform well. 
        """

        values = [str(i).zfill(4), tot_loss_value, time() - grad_step_start_time]
        format_str = 'steps = {0} | overall loss = {1:.3f} | time {2:.3f}'
        logger.info(format_str.format(*values))
        fp.write(values)


def main():
    logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)
    run()


if __name__ == '__main__':
    main()
