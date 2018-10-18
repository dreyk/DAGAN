from dagan_networks_wgan import *
import tensorflow as tf
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
import glob
import logging
import numpy as np

IMAGE_SIZE = (64, 64)

def null_dataset():
    def _input_fn():
        return None

    return _input_fn


def eval_fn():
    inputs = ['0']

    def _input_fn():
        ds = tf.data.Dataset.from_tensor_slices(inputs)

        def _fake_image(_):
            z_vectors = np.random.normal(size=(1,1))
            z = tf.zeros([1, 1, 1, 3], dtype=tf.float32)
            return {'i': z,'j':z,'z':z_vectors}, 0

        ds = ds.map(_fake_image, num_parallel_calls=1)
        return ds

    return _input_fn


def input_fn(params, is_training):
    limit = params['limit_train']
    if limit is None:
        limit = -1

    inputs = []
    i = 0
    for f in glob.iglob(params['data_set'], recursive=True):
        inputs.append(f)
        i += 1
        if (limit > 0) and (i >= limit):
            break
    logging.info('Training files count: {}'.format(len(inputs)))

    def _input_fn():
        ds = tf.data.Dataset.from_tensor_slices(inputs)
        if is_training:
            ds = ds.repeat(count=params['epoch']).shuffle(params['batch_size'] * 10)

        def _read_image(filename):
            image = tf.image.decode_jpeg(tf.read_file(filename), 3)
            image = tf.image.resize_images(image, IMAGE_SIZE)
            image = image / 255.0 - 1
            return image

        def _pair(x):
            z_vectors = np.random.normal(size=(params['batch_size'], params['z_dim']))
            return {'i':x[0],'j':x[1],'z':z_vectors},0

        ds = ds.map(_read_image)

        ds1 = ds.repeat(count=params['epoch']).shuffle(params['batch_size'] * 10)
        ds2 = ds.repeat(count=params['epoch']).shuffle(params['batch_size'] * 10)
        ds = ds.zip((ds1,ds2)).map(_pair)

        if is_training:
            ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
        else:
            ds = ds.padded_batch(params['batch_size'], padded_shapes=(
                {'i': [IMAGE_SIZE[0], IMAGE_SIZE[1], 3],'j': [IMAGE_SIZE[0], IMAGE_SIZE[1], 3]}, tf.TensorShape([])))
        return ds

    return inputs, _input_fn


def _encoder_model_fn(features, labels, mode, params=None, config=None):
    generator_layers = [64, 64, 128, 128]
    discriminator_layers = [64, 64, 128, 128]
    gen_depth_per_layer = params['generator_inner_layers']
    discr_depth_per_layer = params['discriminator_inner_layers']
    gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer]
    discr_inner_layers = [discr_depth_per_layer, discr_depth_per_layer, discr_depth_per_layer,
                          discr_depth_per_layer]
    generator_layer_padding = ["SAME", "SAME", "SAME", "SAME"]

    dagan = DAGAN(batch_size=params['batch_size'], input_x_i=features['i'], input_x_j=features['j'],
                  dropout_rate=params['dropout_rate'], generator_layer_sizes=generator_layers,
                  generator_layer_padding=generator_layer_padding, num_channels=features['i'].shape[3],
                  is_training=(mode == tf.estimator.ModeKeys.TRAIN), augment=params['random_rotate'],
                  discriminator_layer_sizes=discriminator_layers,
                  discr_inner_conv=discr_inner_layers,
                  gen_inner_conv=gen_inner_layers, z_dim=params['z_dim'], z_inputs=features['z'],
                  use_wide_connections=params['use_wide_connections'])

    losses, graph_ops = dagan.init_train()
    if (mode == tf.estimator.ModeKeys.TRAIN):
        accumulated_d_loss = tf.Variable(0.0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        acc_d_loss_op = accumulated_d_loss.assign_add(losses["d_losses"])
        acc_d_loss_zero_op = accumulated_d_loss.assign(tf.zeros_like(accumulated_d_loss))
        accumulated_g_loss = tf.Variable(0.0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        acc_g_loss_op = accumulated_g_loss.assign_add(losses["g_losses"])
        acc_g_loss_zero_op = accumulated_g_loss.assign(tf.zeros_like(accumulated_g_loss))
        reset = tf.group(acc_d_loss_zero_op, acc_g_loss_zero_op)
        d_train = tf.group(graph_ops['d_opt_op'], acc_d_loss_op)
        g_train = tf.group(graph_ops['g_opt_op'], acc_g_loss_op)
        train_hooks = [
            MultiStepOps(params['discriminator_inner_steps'], params['generator_inner_steps'], d_train, g_train, reset)]
        total_loss = accumulated_d_loss + accumulated_g_loss
        train_op = training_util._increment_global_step(1)  # pylint: disable=protected-access
    else:
        total_loss = None
        train_op = None
        train_hooks = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        training_hooks=train_hooks,
        train_op=train_op)


class MultiStepOps(session_run_hook.SessionRunHook):
    def __init__(self, d_steps, g_steps, d_train, g_train, reset):
        self._global_step_tensor = None
        self._d_train = d_train
        self._g_train = g_train
        self._d_steps = d_steps
        self._g_steps = g_steps
        self._reset = reset

    def begin(self):
        None

    def before_run(self, run_context):  # pylint: disable=unused-argument
        for n in range(self._d_steps):
            run_context.session.run([self._d_train])
        for i in range(self._steps):
            run_context.session.run([self._g_train])
        return None

    def after_run(self, run_context, run_values):
        run_context.session.run([self._reset])


class DAGANEstimator(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        def _model_fn(features, labels, mode, params, config):
            return _encoder_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config)

        super(DAGANEstimator, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )
