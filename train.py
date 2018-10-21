import tensorflow as tf
import argparse
import os
import logging
import configparser
import json
import numpy as np
import sys
import dagan_estimator as gan
import PIL.Image
import io
import base64
from mlboardclient.api import client


mlboard = client.Client()

def parse_args():
    conf_parser = argparse.ArgumentParser(
        add_help=False
    )
    conf_parser.add_argument(
        '--checkpoint_dir',
        default=os.environ.get('TRAINING_DIR', 'training') + '/' + os.environ.get('BUILD_ID', '1'),
        help='Directory to save checkpoints and logs',
    )
    args, remaining_argv = conf_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        parents=[conf_parser],
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    checkpoint_dir = args.checkpoint_dir
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    parser.add_argument(
        '--limit_train',
        type=int,
        default=-1,
        help='Limit number files for train. For testing.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size.',
    )
    parser.add_argument(
        '--generator_inner_layers',
        type=int,
        default=1,
        help='generator_inner_layers',
    )

    parser.add_argument(
        '--discriminator_inner_layers',
        type=int,
        default=1,
        help='discriminator_inner_layers',
    )
    parser.add_argument(
        '--generator_inner_steps',
        type=int,
        default=1,
        help='generator_inner_layers',
    )
    parser.add_argument(
        '--discriminator_inner_steps',
        type=int,
        default=5,
        help='discriminator_inner_steps',
    )
    parser.add_argument(
        '--z_dim',
        type=int,
        default=100,
        help='z_dim',
    )
    parser.add_argument(
        '--num_generations',
        type=int,
        default=64,
        help='num_generations',
    )
    parser.add_argument(
        '--test_image',
        type=str,
        default=None,
        help='test_image',
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.5,
        help='dropout_rate',
    )
    parser.add_argument(
        '--random_rotate',
        dest='random_rotate',
        action='store_true',
        help='random_rotate',
    )
    parser.add_argument(
        '--use_wide_connections',
        dest='use_wide_connections',
        action='store_true',
        help='use_wide_connections',
    )


    parser.add_argument(
        '--epoch',
        type=int,
        default=1,
        help='Epoch to trian',
    )
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        default=10,
        help="Log summary every 'save_summary_steps' steps",
    )
    parser.add_argument(
        '--save_checkpoints_secs',
        type=int,
        default=600,
        help="Save checkpoints every 'save_checkpoints_secs' secs.",
    )
    parser.add_argument(
        '--save_checkpoints_steps',
        type=int,
        default=100,
        help="Save checkpoints every 'save_checkpoints_steps' steps",
    )
    parser.add_argument(
        '--keep_checkpoint_max',
        type=int,
        default=5,
        help='The maximum number of recent checkpoint files to keep.',
    )
    parser.add_argument(
        '--log_step_count_steps',
        type=int,
        default=10,
        help='The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.',
    )
    parser.add_argument(
        '--data_set',
        default=None,
        help='Location of training files or evaluation files',
    )
    parser.add_argument(
        '--attr_definition_file',
        default=None,
        help='attr_definition_file',
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(worker=False)
    group.set_defaults(evaluator=False)
    group.set_defaults(test=False)
    group.add_argument('--worker', dest='worker', action='store_true',
                       help='Start in Worker(training) mode.')
    group.add_argument('--evaluator', dest='evaluator', action='store_true',
                       help='Start in evaluation mode')
    group.add_argument('--test', dest='test', action='store_true',
                       help='Test mode')
    p_file = os.path.join(checkpoint_dir, 'parameters.ini')
    if tf.gfile.Exists(p_file):
        parameters = configparser.ConfigParser(allow_no_value=True)
        parameters.read(p_file)
        parser.set_defaults(**dict(parameters.items("PARAMETERS", raw=True)))
    args = parser.parse_args(remaining_argv)
    print('\n*************************\n')
    print(args)
    print('\n*************************\n')
    return checkpoint_dir, args


def test(checkpoint_dir, params):
    logging.info("start test  model")

    session_config = None

    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
        session_config=session_config
    )

    net = gan.DAGANEstimator(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
    )
    input_fn = gan.test_fn(params)
    predictions = net.predict(input_fn)
    num_generations = params['num_generations']
    h = []
    images = []
    for p in predictions:
        p = (p + 1) / 2 * 255
        p = np.uint8(np.clip(p,0,255))
        h.append(p)
        if len(h) == num_generations:
            images.append(np.concatenate(h, axis=1))
            h = []
        if len(images) == num_generations:
            break
    images = np.concatenate(images, axis=0)
    im = PIL.Image.fromarray(images)
    im.save(checkpoint_dir+'/result.png')
    with io.BytesIO() as output:
        im.save(output,format='PNG')
        contents = output.getvalue()
        rpt = '<html><img src="data:image/png;base64,{}"/></html>'.format(base64.b64encode(contents).decode())
        mlboard.update_task_info({'#documents.result.html':rpt})


def train(mode, checkpoint_dir, params):
    logging.info("start build  model")

    save_summary_steps = params['save_summary_steps']
    save_checkpoints_secs = params['save_checkpoints_secs'] if params['save_checkpoints_steps'] is None else None
    save_checkpoints_steps = params['save_checkpoints_steps']

    session_config = None

    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
        save_summary_steps=save_summary_steps,
        save_checkpoints_secs=save_checkpoints_secs,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=params['keep_checkpoint_max'],
        log_step_count_steps=params['log_step_count_steps'],
        session_config=session_config
    )

    net = gan.DAGANEstimator(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
    )
    logging.info("Start %s mode", mode)
    if mode == 'train':
        input_fn = gan.input_fn(params, True)
        net.train(input_fn=input_fn)
    else:
        logging.info("Not implemented")


def main():
    checkpoint_dir, args = parse_args()
    logging.info('------------------')
    logging.info('TF VERSION: {}'.format(tf.__version__))
    logging.info('ARGS: {}'.format(args))
    logging.info('------------------')
    if args.worker:
        mode = 'train'
    elif args.test:
        mode = 'test'
    else:
        mode = 'eval'
        cluster = {'chief': ['fake_worker1:2222'],
                   'ps': ['fake_ps:2222'],
                   'worker': ['fake_worker2:2222']}
        os.environ['TF_CONFIG'] = json.dumps(
            {
                'cluster': cluster,
                'task': {'type': 'evaluator', 'index': 0}
            })

    params = {

        'batch_size': args.batch_size,
        'generator_inner_layers': args.generator_inner_layers,
        'generator_inner_steps': args.generator_inner_steps,
        'discriminator_inner_layers': args.discriminator_inner_layers,
        'discriminator_inner_steps': args.discriminator_inner_steps,
        'save_summary_steps': args.save_summary_steps,
        'save_checkpoints_steps': args.save_checkpoints_steps,
        'save_checkpoints_secs': args.save_checkpoints_secs,
        'keep_checkpoint_max': args.keep_checkpoint_max,
        'log_step_count_steps': args.log_step_count_steps,
        'data_set': args.data_set,
        'epoch': args.epoch,
        'dropout_rate': args.dropout_rate,
        'z_dim': args.z_dim,
        'random_rotate': args.random_rotate,
        'use_wide_connections': args.use_wide_connections,
        'limit_train': args.limit_train,
        'attr_definition_file': args.attr_definition_file,
        'num_generations': args.num_generations,
        'test_image': args.test_image,
    }

    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)

    if args.test:
        test(checkpoint_dir,params)
    else:
        train(mode, checkpoint_dir, params)


if __name__ == '__main__':
    main()
