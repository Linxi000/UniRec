import tensorflow as tf


def embedding(vocab_size,
              embedding_dim,
              zero_pad=True,
              l2_reg=0.0,
              scope='embedding',
              use_reg=True,
              initializer=None,
              reuse=None):

    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        if use_reg is True:
            if initializer is not None and initializer == 'random_normal':
                lookup_table = tf.compat.v1.get_variable(
                    f'{scope}_lookup_table',
                    dtype=tf.float32,
                    shape=[vocab_size, embedding_dim],
                    initializer=tf.random_normal_initializer(
                        0., stddev=1. / (embedding_dim ** 0.5)),
                    regularizer=tf.keras.regularizers.L2(l2_reg))
            else:
                lookup_table = tf.compat.v1.get_variable(
                    f'{scope}_lookup_table',
                    dtype=tf.float32,
                    shape=[vocab_size, embedding_dim],
                    regularizer=tf.keras.regularizers.L2(l2_reg))
        else:
            lookup_table = tf.compat.v1.get_variable(
                f'{scope}_lookup_table',
                dtype=tf.float32,
                shape=[vocab_size, embedding_dim],
                initializer=tf.random_normal_initializer(
                    0., stddev=1. / (embedding_dim ** 0.5)))
        if zero_pad:
            padded_lookup_table = tf.concat((tf.zeros(shape=[1, embedding_dim]),
                                             lookup_table), axis=0)
            return padded_lookup_table, lookup_table
    return lookup_table


def normalize(inputs,
              epsilon=1e-6,
              scope='ln',
              reuse=None):

    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta
    return outputs


def feedforward(inputs,
                num_units=(2048, 512),
                scope='multihead_attention',
                dropout_rate=0.2,
                is_training=True,
                reuse=None,
                normalized=False):
 
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0],
                  "kernel_size": 1, "activation": tf.nn.relu,
                  "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(
            outputs,
            rate=dropout_rate,
            training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1],
                  "kernel_size": 1, "activation": None,
                  "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(
            outputs,
            rate=dropout_rate,
            training=tf.convert_to_tensor(is_training))
        # Residual connection
        if outputs.shape.as_list()[-1] == inputs.shape.as_list()[-1]:
            outputs += inputs
        # Normalize
        if normalized:
            outputs = normalize(outputs)
    return outputs


def add_weight(dimension, name='weight'):
    """
    Create a weight variable
    :param dimension:
    :param name:
    :return:
    """
    with tf.compat.v1.variable_scope(name,
                                     reuse=tf.compat.v1.AUTO_REUSE):
        w = tf.compat.v1.get_variable(
            name=name,
            dtype=tf.float32,
            shape=[dimension, dimension],
            initializer=tf.random_normal_initializer(
                0., stddev=1. / (dimension ** 0.5)))
    return w

def linear_layer(input, output_dim, trainable=True, name='linear_layer'):
    input_dim = int(input.get_shape()[-1])  

    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
       
        weights = tf.compat.v1.get_variable('weights', shape=[input_dim, output_dim],
                                  initializer=tf.random_normal_initializer(0., stddev=1. / (input_dim ** 0.5)),
                                  trainable=trainable)
        biases = tf.compat.v1.get_variable('biases', shape=[output_dim],
                                 initializer=tf.constant_initializer(0.0),
                                 trainable=trainable)

        output = tf.matmul(input, weights) + biases

    return output

