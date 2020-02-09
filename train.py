"""Train the built ResNet model."""
import numpy as np
import tensorflow as tf
from resnet50 import ResNet


class Train:
    def __init__(self, train_x, train_y, valid_x=None, valid_y=None,
                 batch_size=10, learning_rate=0.001, num_epochs=200,
                 save_model=False):
        """__init__ Constructor.

        Parameters
        ----------
        train_x : tf tensor
            The input training data.
        train_y : tf tensor
            The training data classes/labels.
        valid_x : tensor
            The validation data.
        valid_y : tensor
            The validation data classes.
        batch_size : int
            The size of the batch.
        learning_rate : float
            The learning rate.
        num_epochs : int
            The number of epochs.
        save_model : bool
            Whether to save the model or not.

        Returns
        -------
        None

        """
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.format_size = [64, 64]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_model = save_model

    def train_model(self):
        assert len(self.train_x.shape) == 4
        [num_images, img_height, img_width, num_channels] = self.train_x.shape
        num_classes = self.train_y.shape[-1]
        num_steps = int(np.ceil(num_images / float(self.batch_size)))

        tf.reset_default_graph()
        # build the graph and define objective function
        graph = tf.Graph()
        with graph.as_default():
            # build graph
            train_maps_raw = tf.placeholder(
                tf.float32, [None, img_height, img_width, num_channels])
            train_maps = tf.image.resize_images(
                train_maps_raw, self.format_size)
            train_labels = tf.placeholder(tf.float32, [None, num_classes])
            resnet = ResNet()
            Y_hat, model_params = resnet.build_model()
            Z = model_params['out']['Z']

            # loss function
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=Z,
                labels=train_labels)
            loss = tf.reduce_mean(cross_entropy)

            # optimizer
            optimizer = tf.train.AdamOptimizer(
                self.learning_rate).minimize(loss)

        # saver to save the trained model
        saver = tf.train.Saver()

        # Set configs
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # train the graph
        with tf.Session(graph=graph, config=config) as session:
            session.run(tf.initialize_all_variables())
            try:
                for epoch in range(self.num_epochs):
                    for step in range(num_steps):
                        offset = (
                            step * self.batch_size) % (num_images -
                                                       self.batch_size)
                        batch_data = self.train_x[offset:(
                            offset + self.batch_size), :, :, :]
                        batch_labels = self.train_y[offset:(
                            offset + self.batch_size), :]
                        feed_dict = {train_maps_raw: batch_data,
                                     train_labels: batch_labels}
                        _, l, predictions = session.run(
                            [optimizer, loss], feed_dict=feed_dict)
                    print('Epoch %2d/%2d:\n\t'
                          'Train Loss = %.2f\t '
                          'Accuracy = %.2f%%'
                          % (epoch+1, self.num_epochs, l,
                             self.accuracy(predictions, batch_labels)))
            except Exception as e:
                print('exception:', e)
            if (self.save_model):
                # save_path = saver.save(session, 'model.tensorflow')
                save_path = saver.save(session, 'model/VGG16_modelParams.ckpt')
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
