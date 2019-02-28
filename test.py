# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Validate mobilenet_v1 with options for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import math
import numpy as np
import tensorflow as tf
from split_char import split_char

import mobilenet_v1
import preprocessing_factory
import os

tf.logging.set_verbosity(tf.logging.INFO)
slim = tf.contrib.slim

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('num_classes', 36, 'Number of classes to distinguish')
flags.DEFINE_integer('image_size', 128, 'Input image resolution')
flags.DEFINE_float('depth_multiplier', 0.75, 'Depth multiplier for mobilenet')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('checkpoint_path', 'model_dis1/', 'The directory for checkpoints')
flags.DEFINE_string('eval_dir', '', 'Directory for writing eval event logs')
flags.DEFINE_string('train_dir', '', 'Location of training checkpoints')
flags.DEFINE_string('dataset_dir', '', 'Locatizon of dataset')
flags.DEFINE_string('checkpoint_exclude_scopes', '', 'not loading these layers')
flags.DEFINE_string('ignore_missing_vars', '', 'ignore the missing variables')
flags.DEFINE_string('test_pic', 'test1.png', 'path of test image')
FLAGS = flags.FLAGS

names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
         'U', 'V', 'W', 'X', 'Y', 'Z']


# the picture is one channel and you have to change to a three channel image.
def to3channel(image_data):
    for i in range(len(image_data)):
        image = image_data[i] / 255.0
        images = np.stack((image, image, image), axis=-1)
        image_data[i] = images


# print all tensor, used to debug
def print_tensor(sess):
    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)

    for var, val in zip(tvars, tvars_vals):
        print(var.name, val)


def run():
    """Build the mobilenet_v1 model for evaluation.
    Returns:
      g: graph with rewrites after insertion of quantization ops and batch norm
      folding.
      eval_ops: eval ops for inference.
      variables_to_restore: List of variables to restore from checkpoint.
    """
    image_datas = split_char(FLAGS.test_pic)
    to3channel(image_datas)
    build_model(image_datas)


def build_model(image_datas):
    with tf.Graph().as_default():

        image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])

        def preprocessing_fn(image, output_height, output_width, **kwargs):
            return preprocessing_factory.preprocess_image(
                image, output_height, output_width, is_training=False, **kwargs)

        processed_image = preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
        processed_images = tf.expand_dims(processed_image, 0)

        scope = mobilenet_v1.mobilenet_v1_arg_scope(
            weight_decay=0.0)
        # Create the model, use the default arg scope to configure the batch norm parameters.

        with slim.arg_scope(scope):
            # 1000 classes instead of 1001.
            logits, _ = mobilenet_v1.mobilenet_v1(
                processed_images,
                is_training=False,
                depth_multiplier=FLAGS.depth_multiplier,
                num_classes=FLAGS.num_classes)
        probabilities = tf.nn.softmax(logits)
        variables_to_restore = slim.get_variables_to_restore()

        saver = tf.train.Saver(var_list=variables_to_restore)
        with tf.Session() as sess:

            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))
            # print_tensor(sess)
            for image_data in image_datas:
                p_image, g_probabilities = sess.run([processed_image, probabilities], feed_dict={image: image_data})

                g_probabilities = g_probabilities[0].squeeze()
                sorted_inds = [i[0] for i in sorted(enumerate(-g_probabilities), key=lambda x: x[1])]

                for i in range(len(sorted_inds)):
                    index = sorted_inds[i]
                    # print the probabilities of each category.
                    # print('Probability %0.2f%% => [%s]' % (g_probabilities[index] * 100, names[index]))
                print('recognize character:', names[sorted_inds[0]])
                cv2.imshow('1', p_image)
                cv2.waitKey()


def main(unused_arg):
    run()


if __name__ == '__main__':
    tf.app.run(main)
