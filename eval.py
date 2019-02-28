from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

import font
import mobilenet_v1
import preprocessing_factory


slim = tf.contrib.slim

flags = tf.app.flags

flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_string('checkpoint_dir', 'model_dis1\\model.ckpt-30757',
                    'Directory for writing training checkpoints and logs')
flags.DEFINE_string('checkpoint_exclude_scopes', '',
                    'not loading these layers')
flags.DEFINE_string('dataset_dir', 'data', 'Location of dataset')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('num_classes', 36, 'Number of classes to distinguish')
flags.DEFINE_integer('number_of_steps', None,
                     'Number of training steps to perform before stopping')
flags.DEFINE_integer('image_size', 128, 'Input image resolution')
flags.DEFINE_float('depth_multiplier', 0.75, 'Depth multiplier for mobilenet')
flags.DEFINE_integer('num_examples', 36*216, 'Number of examples to evaluate')
flags.DEFINE_string('eval_dir', 'mobilenet_0.75_128/', 'Directory for writing eval event logs')


FLAGS = flags.FLAGS


def imagenet_input(is_training):
    """Data reader for imagenet.
    Reads in imagenet data and performs pre-processing on the images.
    Args:
       is_training: bool specifying if train or validation dataset is needed.
    Returns:
       A batch of images and labels.
    """
    if is_training:
        dataset = font.get_split('train', FLAGS.dataset_dir)
    else:
        dataset = font.get_split('validation', FLAGS.dataset_dir)

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=is_training,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])

    def preprocessing_fn(image, output_height, output_width, **kwargs):
        return preprocessing_factory.preprocess_image(
            image, output_height, output_width, is_training=is_training, **kwargs)

    image = preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=4,
        capacity=5 * FLAGS.batch_size)
    labels = slim.one_hot_encoding(labels, FLAGS.num_classes)
    return images, labels


def metrics(logits, labels):
    """Specify the metrics for eval.
    Args:
      logits: Logits output from the graph.
      labels: Ground truth labels for inputs.
    Returns:
       Eval Op for the graph.
    """
    labels = tf.squeeze(labels)
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': tf.metrics.accuracy(tf.argmax(logits, 1), tf.argmax(labels, 1)),
    })
    for name, value in names_to_values.items():
        slim.summaries.add_scalar_summary(
            value, name, prefix='eval', print_summary=True)
    return list(names_to_updates.values())


def build_model():
    """Build the mobilenet_v1 model for evaluation.
    Returns:
      g: graph with rewrites after insertion of quantization ops and batch norm
      folding.
      eval_ops: eval ops for inference.
      variables_to_restore: List of variables to restore from checkpoint.
    """
    g = tf.Graph()
    with g.as_default():
        inputs, labels = imagenet_input(is_training=False)

        scope = mobilenet_v1.mobilenet_v1_arg_scope(
            is_training=False, weight_decay=0.0)
        with slim.arg_scope(scope):
            logits, _ = mobilenet_v1.mobilenet_v1(
                inputs,
                is_training=False,
                depth_multiplier=FLAGS.depth_multiplier,
                num_classes=FLAGS.num_classes)

        if FLAGS.quantize:
            tf.contrib.quantize.create_eval_graph()

        eval_ops = metrics(logits, labels)

    return g, eval_ops


def eval_model():
    """Evaluates mobilenet_v1."""
    g, eval_ops = build_model()
    with g.as_default():
        num_batches = math.ceil(FLAGS.num_examples / float(FLAGS.batch_size))
        slim.evaluation.evaluate_once(
            FLAGS.master,
            FLAGS.checkpoint_dir,
            logdir=FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=eval_ops)


def main(unused_arg):
    eval_model()


if __name__ == '__main__':
    tf.app.run(main)
