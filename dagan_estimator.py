from dagan_networks_wgan import *
import tensorflow as tf
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
import glob
import logging
import numpy as np
import pandas as pd
import PIL.Image

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
            z_vectors = np.random.normal(size=(1, 1))
            z = tf.zeros([1, 1, 1, 3], dtype=tf.float32)
            return {'i': z, 'j': z, 'z': z_vectors}, 0

        ds = ds.map(_fake_image, num_parallel_calls=1)
        return ds

    return _input_fn


def input_fn(params, is_training):
    limit = params['limit_train']
    if limit is None:
        limit = -1

    files = pd.read_csv(params['attr_definition_file'])
    if limit < 1:
        limit = len(files)
    class_files = []
    for c in files.columns.values:
        if c == 'image_id':
            continue
        v = files.loc[files[c] == 1]['image_id'].values
        class_files.append(v)
    z = np.zeros((params['batch_size'],params['z_dim']), dtype=np.float32)
    limit = limit*params['epoch']//params['batch_size']+1
    def _input_fn():
        def _gen():
            for _ in range(limit):
                sc = np.random.choice(len(class_files),params['batch_size'])
                a_batch = []
                b_batch = []
                for i in len(params['batch_size']):
                    samples = np.random.choice(class_files[sc[i]], 2)
                    im = PIL.Image.open(samples[0])
                    im = im.resize((IMAGE_SIZE[0], IMAGE_SIZE[1]))
                    a = np.asarray(im)
                    im = PIL.Image.open(samples[1])
                    im = im.resize((IMAGE_SIZE[0], IMAGE_SIZE[1]))
                    b = np.asarray(im)
                    a_batch.append(a)
                    b_batch.append(b)
                a_batch = np.stack(a_batch)/127.5-1
                b_batch = np.stack(b_batch)/127.5-1
            yield ({'i': a_batch, 'j': b_batch, 'z': z}, 0)

        cshape = tf.TensorShape([params['batch_size'], IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]])
        ds = tf.data.Dataset.from_generator(_gen, ({'i': tf.float32, 'j': tf.float32, 'z': tf.float32}, tf.int32), (
        {'i': cshape, 'j': cshape, 'z': tf.TensorShape([params['batch_size'], params['z_dim']])},
        tf.TensorShape([None])))
        return ds

    return _input_fn


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
                  is_training=(mode == tf.estimator.ModeKeys.TRAIN),
                  augment=tf.constant(params['random_rotate'], dtype=tf.bool),
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
        for n in range(self._g_steps):
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
