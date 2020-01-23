import numpy as np
import tensorflow as tf


class ResNet:
    def __init__(self):
        pass

    def get_weights(self, shape, name):
        return tf.get_variable(name, shape=shape)

    def get_bias(self, shape, name):
        return tf.zeros(shape=shape, name=name)

    def zero_padding(self, tensor, pad=(3, 3)):
        paddings = tf.constant([[0, 0], [pad[0], pad[0]],
                                [pad[1], pad[1]], [0, 0]])
        return tf.pad(tensor, paddings, 'CONSTANT')

    def flatten(self, tensor):
        return tf.contrib.layers.flatten(tensor)

    def dense(self, tensor, out, name):
        in_prev = tensor.shape.as_list()[1]
        W = self.get_weights((in_prev, out), name=name+'_W')
        b = self.get_bias((1, out), name=name+'_b')
        Z = tf.add(tf.matmul(tensor, W), b, name=name+'_Z')
        A = tf.nn.softmax(Z, name=name)
        params = {'W': W, 'b': b, 'Z': Z, 'A': A}
        return A, params

    def conv2D(self, A_prev, filters, k_size, strides, padding, name):
        m, in_H, in_W, in_C = A_prev.shape.as_list()

        w_shape = (k_size[0], k_size[1], in_C, filters)
        b_shape = (1, 1, 1, filters)

        W = self.get_weights(shape=w_shape, name=name+'_W')
        b = self.get_bias(shape=b_shape, name=name+'_b')

        strides = [1, strides[0], strides[1], 1]

        A = tf.nn.conv2d(A_prev, W, strides=strides,
                         padding=padding, name=name)
        params = {'W': W, 'b': b, 'A': A}
        return A, params

    def batch_norm(self, tensor, name):
        m_, v_ = tf.nn.moments(tensor, axes=[0, 1, 2], keep_dims=False)
        beta_ = tf.zeros(tensor.shape.as_list()[3])
        gamma_ = tf.ones(tensor.shape.as_list()[3])
        bn = tf.nn.batch_normalization(tensor, mean=m_, variance=v_,
                                       offset=beta_, scale=gamma_,
                                       variance_epsilon=1e-4)
        return bn


def identity_block(self, tensor, f, filters, stage, block):
    """
    Implementing a ResNet identity block with shortcut path
    passing over 3 Conv Layers
    @params
    X - input tensor of shape (m, in_H, in_W, in_C)
    f - size of middle layer filter
    filters - tuple of number of filters in 3 layers
    stage - used to name the layers
    block - used to name the layers
    @returns
    A - Output of identity_block
    params - Params used in identity block
    """

    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'

    l1_f, l2_f, l3_f = filters

    params = {}

    A1, params[conv_name+'2a'] = \
        self.conv2D(tensor, filters=l1_f, k_size=(1, 1), strides=(1, 1),
                    padding='VALID', name=conv_name+'2a')
    A1_bn = self.batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)
    params[conv_name+'2a']['bn'] = A1_bn
    params[conv_name+'2a']['act'] = A1_bn

    A2, params[conv_name+'2b'] = \
        self.conv2D(A1_act, filters=l2_f, k_size=(f, f), strides=(1, 1),
                    padding='SAME', name=conv_name+'2b')
    A2_bn = self.batch_norm(A2, name=bn_name+'2b')
    A2_act = tf.nn.relu(A2_bn)
    params[conv_name+'2b']['bn'] = A2_bn
    params[conv_name+'2b']['act'] = A2_act

    A3, params[conv_name+'2c'] = \
        self.conv2D(A2_act, filters=l3_f, k_size=(1, 1), strides=(1, 1),
                    padding='VALID', name=conv_name+'2c')
    A3_bn = self.batch_norm(A3, name=bn_name+'2c')

    A3_add = tf.add(A3_bn, tensor)
    A = tf.nn.relu(A3_add)
    params[conv_name+'2c']['bn'] = A3_bn
    params[conv_name+'2c']['add'] = A3_add
    params['out'] = A
    return A, params


def convolutional_block(self, tensor, f, filters, stage, block, s=2):
    """
    Implementing a ResNet convolutional block with shortcut path
    passing over 3 Conv Layers having different sizes
    @params
    X - input tensor of shape (m, in_H, in_W, in_C)
    f - size of middle layer filter
    filters - tuple of number of filters in 3 layers
    stage - used to name the layers
    block - used to name the layers
    s - strides used in first layer of convolutional block
    @returns
    A - Output of convolutional_block
    params - Params used in convolutional block
    """

    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'

    l1_f, l2_f, l3_f = filters

    params = {}

    A1, params[conv_name+'2a'] = \
        self.conv2D(tensor, filters=l1_f, k_size=(1, 1), strides=(s, s),
                    padding='VALID', name=conv_name+'2a')
    A1_bn = self.batch_norm(A1, name=bn_name+'2a')
    A1_act = tf.nn.relu(A1_bn)
    params[conv_name+'2a']['bn'] = A1_bn
    params[conv_name+'2a']['act'] = A1_bn

    A2, params[conv_name+'2b'] = \
        self.conv2D(A1_act, filters=l2_f, k_size=(f, f), strides=(1, 1),
                    padding='SAME', name=conv_name+'2b')
    A2_bn = self.batch_norm(A2, name=bn_name+'2b')
    A2_act = tf.nn.relu(A2_bn)
    params[conv_name+'2b']['bn'] = A2_bn
    params[conv_name+'2b']['act'] = A2_act

    A3, params[conv_name+'2c'] = \
        self.conv2D(A2_act, filters=l3_f, k_size=(1, 1), strides=(1, 1),
                    padding='VALID', name=conv_name+'2c')
    A3_bn = self.batch_norm(A3, name=bn_name+'2c')
    params[conv_name+'2c']['bn'] = A3_bn

    A_, params[conv_name+'1'] = \
        self.conv2D(tensor, filters=l3_f, k_size=(1, 1), strides=(s, s),
                    padding='VALID', name=conv_name+'1')
    A_bn_ = self.batch_norm(A_, name=bn_name+'1')

    A3_add = tf.add(A3_bn, A_bn_)
    A = tf.nn.relu(A3_add)
    params[conv_name+'2c']['add'] = A3_add
    params[conv_name+'1']['bn'] = A_bn_
    params['out'] = A
    return A, params
