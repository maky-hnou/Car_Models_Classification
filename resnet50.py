"""Build the ResNet50 model."""
import tensorflow as tf


class ResNet:
    """Build the ResNet CNN.

    Parameters
    ----------
    input_shape : list
        The shape of the input data.
    categories : int
        The number of categories.

    Attributes
    ----------
    input_shape
    categories

    """

    def __init__(self, input_shape=[64, 64, 3], categories=2):
        """Initialize the variables.

        Parameters
        ----------
        input_shape : list
            The shape of the input data.
        categories : int
            The number of categories.

        Returns
        -------
        None

        """
        self.input_shape = input_shape
        self.categories = categories

    def get_weights(self, shape, name):
        """Create weights.

        Parameters
        ----------
        shape : tuple
            The shape of the weights.
        name : str
            The name of the weights.

        Returns
        -------
        tf tensor
            A tensor containing the weights.

        """
        return tf.get_variable(name, shape=shape)

    def get_bias(self, shape, name):
        """Create bias.

        Parameters
        ----------
        shape : tuple
            The shape of the bias.
        name : str
            The name of the bias.

        Returns
        -------
        tf tensor
            A tensor containing the bias.

        """
        return tf.zeros(shape=shape, name=name)

    def zero_padding(self, tensor, pad=(3, 3)):
        """Pad an input tensor.

        Parameters
        ----------
        tensor : tf tensor
            The tensor to be padded.
        pad : tuple
            The shape of the padding tensor.

        Returns
        -------
        tf tensor
            A padded tensor.

        """
        paddings = tf.constant([[0, 0], [pad[0], pad[0]],
                                [pad[1], pad[1]], [0, 0]])
        return tf.pad(tensor, paddings, 'CONSTANT')

    def flatten(self, tensor):
        """Flatten an input tensor.

        Parameters
        ----------
        tensor : tf tensor
            The input tensor to be flattened.

        Returns
        -------
        tf tensor
            The flattened tensor.

        """
        return tf.contrib.layers.flatten(tensor)

    def dense(self, tensor, out, name):
        """Create a Fully Connected Layer.

        Parameters
        ----------
        tensor : tf tensor
            The input tensor.
        out : int
            The number of categories.
        name : str
            The name of the output tensor.

        Returns
        -------
        tf tensor
            The output of the Fully Connected Layer.

        """
        in_prev = tensor.shape.as_list()[1]
        W = self.get_weights((in_prev, out), name=name+'_W')
        b = self.get_bias((1, out), name=name+'_b')
        Z = tf.add(tf.matmul(tensor, W), b, name=name+'_Z')
        A = tf.nn.softmax(Z, name=name)
        params = {'W': W, 'b': b, 'Z': Z, 'A': A}
        return A, params

    def conv2D(self, A_prev, filters, k_size, strides, padding, name):
        """Apply 2D convolution.

        Parameters
        ----------
        A_prev : tf tensor
            The input tensor.
        filters : tuple
            A tuple of the number of filters per layer.
        k_size : tuple
            The size of the kernel.
        strides : tuple
            The number of strides.
        padding : str
            The type of padding algorithm to use.
        name : str
            A name for the operation.

        Returns
        -------
        tf tensor
            The convolved tensor.
        dict
            The dictionary that contains the params.

        """
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
        """Normalize an input tensor.

        Parameters
        ----------
        tensor : tf tensor
            The input tensor.
        name : str
            A name for the operation.

        Returns
        -------
        tf tensor
            The normalized tensor.

        """
        m_, v_ = tf.nn.moments(tensor, axes=[0, 1, 2], keep_dims=False)
        beta_ = tf.zeros(tensor.shape.as_list()[3])
        gamma_ = tf.ones(tensor.shape.as_list()[3])
        bn = tf.nn.batch_normalization(tensor, mean=m_, variance=v_,
                                       offset=beta_, scale=gamma_,
                                       variance_epsilon=1e-4, name=name)
        return bn


def identity_block(self, tensor, f, filters, stage, block):
    """Implement identity block with shortcut path passing over 3 Conv Layers.

    Parameters
    ----------
    tensor : tf tensor
        The input tensor of shape (m, in_H, in_W, in_C).
    f : int
        The size of middle layer filter.
    filters : tuple
        The number of filters in 3 layers.
    stage : str
        Used to name the layers.
    block : str
        Used to name the layers.

    Returns
    -------
    tf tensor
        The output of identity block.
    dict
        The dictionary of parameters.

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
        """Implement ResNet conv block with shortcut path over 3 Conv Layers.

        Parameters
        ----------
        tensor : tf tensor
            The input tensor of shape (m, in_H, in_W, in_C).
        f : int
            The size of middle layer filter.
        filters : tuple
            The number of filters in 3 layers.
        stage : str
            Used to name the layers.
        block : str
            Used to name the layers.

        Returns
        -------
        tf tensor
            The output of identity block.
        dict
            The dictionary of parameters.

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

    def ResNet50(self):

        input_shape = [None] + self.input_shape
        params = {}

        X_input = tf.placeholder(
            tf.float32, shape=input_shape, name='input_layer')

        X = self.zero_padding(X_input, (3, 3))
        params['input'] = X_input
        params['zero_pad'] = X

        # Stage 1
        params['stage1'] = {}
        A_1, params['stage1']['conv'] = \
            self.conv2D(X, filters=64, k_size=(7, 7), strides=(2, 2),
                        padding='VALID', name='conv1')
        A_1_bn = self.batch_norm(A_1, name='bn_conv1')
        A_1_act = tf.nn.relu(A_1_bn)
        A_1_pool = tf.nn.max_pool(A_1_act, ksize=(1, 3, 3, 1),
                                  strides=(1, 2, 2, 1),
                                  padding='VALID')
        params['stage1']['bn'] = A_1_bn
        params['stage1']['act'] = A_1_act
        params['stage1']['pool'] = A_1_pool

        # Stage 2
        params['stage2'] = {}
        A_2_cb, params['stage2']['cb'] = \
            self.convolutional_block(A_1_pool, f=3, filters=[64, 64, 256],
                                     stage=2, block='a', s=1)
        A_2_ib1, params['stage2']['ib1'] = \
            self.identity_block(A_2_cb, f=3, filters=[64, 64, 256],
                                stage=2, block='b')
        A_2_ib2, params['stage2']['ib2'] = \
            self.identity_block(A_2_ib1, f=3, filters=[64, 64, 256],
                                stage=2, block='c')

        # Stage 3
        params['stage3'] = {}
        A_3_cb, params['stage3']['cb'] = \
            self.convolutional_block(A_2_ib2, 3, [128, 128, 512],
                                     stage=3, block='a', s=2)
        A_3_ib1, params['stage3']['ib1'] = \
            self.identity_block(A_3_cb, 3, [128, 128, 512],
                                stage=3, block='b')
        A_3_ib2, params['stage3']['ib2'] = \
            self.identity_block(A_3_ib1, 3, [128, 128, 512],
                                stage=3, block='c')
        A_3_ib3, params['stage3']['ib3'] = \
            self.identity_block(A_3_ib2, 3, [128, 128, 512],
                                stage=3, block='d')

        # Stage 4
        params['stage4'] = {}
        A_4_cb, params['stage4']['cb'] = \
            self.convolutional_block(A_3_ib3, 3, [256, 256, 1024],
                                     stage=4, block='a', s=2)
        A_4_ib1, params['stage4']['ib1'] = \
            self.identity_block(A_4_cb, 3, [256, 256, 1024],
                                stage=4, block='b')
        A_4_ib2, params['stage4']['ib2'] = \
            self.identity_block(A_4_ib1, 3, [256, 256, 1024],
                                stage=4, block='c')
        A_4_ib3, params['stage4']['ib3'] = \
            self.identity_block(A_4_ib2, 3, [256, 256, 1024],
                                stage=4, block='d')
        A_4_ib4, params['stage4']['ib4'] = \
            self.identity_block(A_4_ib3, 3, [256, 256, 1024],
                                stage=4, block='e')
        A_4_ib5, params['stage4']['ib5'] = \
            self.identity_block(A_4_ib4, 3, [256, 256, 1024],
                                stage=4, block='f')

        # Stage 5
        params['stage5'] = {}
        A_5_cb, params['stage5']['cb'] = \
            self.convolutional_block(A_4_ib5, 3, [512, 512, 2048],
                                     stage=5, block='a', s=2)
        A_5_ib1, params['stage5']['ib1'] = \
            self.identity_block(A_5_cb, 3, [512, 512, 2048],
                                stage=5, block='b')
        A_5_ib2, params['stage5']['ib2'] = \
            self.identity_block(A_5_ib1, 3, [512, 512, 2048],
                                stage=5, block='c')

        # Average Pooling
        A_avg_pool = tf.nn.avg_pool(A_5_ib2, ksize=(1, 2, 2, 1),
                                    strides=(1, 2, 2, 1),
                                    padding='VALID', name='avg_pool')
        params['avg_pool'] = A_avg_pool

        # Output Layer
        A_flat = self.flatten(A_avg_pool)
        params['flatten'] = A_flat
        A_out, params['out'] = self.dense(
            A_flat, self.categories, name='fc'+str(self.categories))

        return A_out, params
