"""Train the ResNet model."""
import os

import numpy as np
import tensorflow as tf
from resnet50 import ResNet


class TrainModel:
    """Train the model.

    Parameters
    ----------
    train_x : tf tensor
        The input training data.
    train_y : tf tensor
        The training data classes.
    learning_rate : float
        The learning rate.
    num_epochs : int
        The number of epochs.
    batch_size : int
        The size of the batch.
    save_model : bool
        Whether to save the model or not.

    Attributes
    ----------
    train_x
    train_y
    learning_rate
    num_epochs
    batch_size
    save_model

    """

    def __init__(self, train_x, train_y, learning_rate=0.001,
                 num_epochs=5000, batch_size=8, save_model=True):
        self.train_x = train_x
        self.train_y = train_y
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_model = save_model

    def train(self):
        """Train the model and save the weights.

        Returns
        -------
        None

        """
        assert len(self.train_x.shape) == 4
        num_images = self.train_x.shape[0]
        num_steps = int(np.ceil(num_images / float(self.batch_size)))

        resnet = ResNet()
        Y_hat, model_params = resnet.build_model()
        X = model_params['input']
        Y_true = tf.placeholder(dtype=tf.float32, shape=[None, 196])
        Z = model_params['out']['Z']

        # loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Z,
                                                                labels=Y_true)
        loss_function = tf.reduce_mean(cross_entropy)
        # optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(
            self.learning_rate).minimize(loss_function)
        # saver to save the trained model
        saver = tf.compat.v1.train.Saver()

        # Set configs
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # train the graph
        with tf.compat.v1.Session(config=config) as session:
            session.run(tf.initialize_all_variables())
            for epoch in range(self.num_epochs):
                for step in range(num_steps):
                    offset = (
                        step * self.batch_size) % (num_images -
                                                   self.batch_size)
                    batch_data = self.train_x[offset:(
                        offset + self.batch_size), :, :, :]
                    batch_labels = self.train_y[offset:(
                        offset + self.batch_size), :]
                    feed_dict = {X: batch_data,
                                 Y_true: batch_labels}
                    _, loss = session.run(
                        [optimizer, loss_function], feed_dict=feed_dict)
                print('Epoch %2d/%2d:\n\tTrain Loss = %.2f\t'
                      % (epoch+1, self.num_epochs, loss))
            if (self.save_model):
                os.makedirs('./model/')
                # save_path = saver.save(session, 'model.tensorflow')
                save_path = saver.save(session, 'model/ResNet50_model.ckpt')
                print('The model has been saved to ' + save_path)
            session.close()

    def accuracy(self, predictions, labels):
        """Calculate the accuracy.

        Parameters
        ----------
        predictions : numpy ndarray
            An array containing the predictions probabilities.
        labels : numpy ndarray
            Description of parameter `labels`.

        Returns
        -------
        float
            The Calculated accuracy.

        """
        return (100.0 * np.sum(np.argmax(predictions, 1) ==
                               np.argmax(labels, 1)) / predictions.shape[0])
