import os
import tensorflow as tf

from unirec.logging import get_logger
from unirec.utils.params import process_params
from unirec.data.datasets import dataset_factory
from unirec.models import ModelFactory
from unirec.train.trainer import Trainer


def entrypoint(params):
    """ Command entrypoint
    :param params: Deserialized JSON configuration file
                   provided in CLI args.
    """
    logger = get_logger()

    training_params, model_params = process_params(params)

    if not os.path.exists(training_params['model_dir']):
        os.makedirs(training_params['model_dir'], exist_ok=True)
    logger.info(training_params['model_dir'])

    data = dataset_factory(params=params)

    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    with tf.compat.v1.Session(config=sess_config) as sess:

        model = ModelFactory.generate_model(sess=sess,
                                            params=training_params,
                                            n_users=data['n_users'],
                                            n_items=data['n_items'])

        sess.run(tf.compat.v1.global_variables_initializer())

        trainer = Trainer(sess, model, params)

        logger.info('Start model training')
        trainer.fit(data=data)
        logger.info('Model training done')
