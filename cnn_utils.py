"""Utils functions."""

import tensorflow as tf

# construct a Convolutional Layer


def convolution_layer(layer_name, input_maps, num_output_channels,
                      kernel_size=[3, 3], stride=[1, 1, 1, 1]):
    """Create a Convolutional layer.

    Parameters
    ----------
    layer_name : str
        The name of the layer.
    input_maps : tf tensor
        The input data.
    num_output_channels : int
        The number of color channels.
    kernel_size : list
        The shape of the filter.
    stride : list
        The stride of the sliding window for each dimension of input_maps.

    Returns
    -------
    tf tensors
        The outputs of the convolutional layer.

    """
    num_input_channels = input_maps.get_shape()[-1].value
    with tf.name_scope(layer_name) as scope:
        kernel = tf.get_variable(
            scope+'W', shape=[kernel_size[0], kernel_size[1],
                              num_input_channels, num_output_channels],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        convolution = tf.nn.conv2d(
            input_maps, kernel, stride, padding='SAME')
        bias = tf.Variable(
            tf.constant(
                0.0, shape=[num_output_channels], dtype=tf.float32),
            trainable=True, name='b')
        output = tf.nn.relu(tf.nn.bias_add(convolution, bias), name=scope)
        return output, kernel, bias

# Construct a Max Pooling Layer


def max_pooling_layer(layer_name, input_maps,
                      kernel_size=[2, 2], stride=[1, 2, 2, 1]):
    """Perform the max pooling on the input.

    Parameters
    ----------
    layer_name : str
        The name of the layer.
    input_maps : tf tensor
        The input data.
    kernel_size : list
        The shape of the filter.
    stride : list
        The stride of the sliding window for each dimension of input_maps.

    Returns
    -------
    tf tensor
        The max pooled output tensor.

    """
    output = tf.nn.max_pool(input_maps,
                            ksize=[1, kernel_size[0], kernel_size[1], 1],
                            strides=stride,
                            padding='SAME',
                            name=layer_name)
    return output

# Construct an Average Pooling Layer


def avg_pooling_layer(layer_name, input_maps,
                      kernel_size=[2, 2], stride=[1, 2, 2, 1]):
    """Perform the average pooling on the input.

    Parameters
    ----------
    layer_name : str
        The name of the layer.
    input_maps : tf tensor
        The input data.
    kernel_size : list
        The shape of the filter.
    stride : list
        The stride of the sliding window for each dimension of input_maps.

    Returns
    -------
    tf tensor
        The max pooled output tensor.

    """
    output = tf.nn.avg_pool(input_maps,
                            ksize=[1, kernel_size[0], kernel_size[1], 1],
                            strides=stride,
                            padding='SAME',
                            name=layer_name)
    return output

# Construct a Fully Connected Layer


def fully_connected_layer(layer_name, input_maps, num_output_nodes):
    """Create a fully connected layer.

    Parameters
    ----------
    layer_name : str
        The name of the layer.
    input_maps : tf tensor
        The input data.
    num_output_nodes : int
        The number of the output nodes.

    Returns
    -------
    tf tensors
        The outputs of the fully connected layer.

    """
    shape = input_maps.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value
    with tf.name_scope(layer_name) as scope:
        kernel = tf.get_variable(
            scope+'W', shape=[size, num_output_nodes],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.Variable(
            tf.constant(
                0.1, shape=[num_output_nodes], dtype=tf.float32),
            trainable=True, name='b')
        flat = tf.reshape(input_maps, [-1, size])
        output = tf.nn.relu(tf.nn.bias_add(tf.matmul(flat, kernel), bias))
        return output, kernel, bias
