"""Test the model."""

import cv2
import numpy as np
import tensorflow as tf
from resnet50 import ResNet


class Test:
    """Run the cars classification.

    Parameters
    ----------
    image_path : str
        The path to the input image.
    graph_path : str
        The path to the saved model.

    Attributes
    ----------
    image_size : tuple[int]
        The image dimensions.
    image_path
    graph_path

    """
    def __init__(self, image_path, graph_path):
        self.image_path = image_path
        self.graph_path = graph_path
        self.image_size = (64, 64)

    def prepare_image(self):
        """Return a resized image.

        Returns
        -------
        numpy ndarray[int]
            The resied image.

        """
        img = cv2.imread(self.image_path)
        resized = cv2.resize(img, self.image_size)
        return resized

    def classify(self):
        """Return the index of the car model.

        Returns
        -------
        int
            The category index.

        """
        resnet = ResNet()
        Y_hat, model_params = resnet.build_model()
        X = model_params['input']
        saver = tf.train.Saver()
        image = self.prepare_image(self.image_path)
        with tf.Session() as session:
            saver.restore(session, self.graph_path)
            category = session.run(Y_hat, feed_dict={X: [image]})

        return np.argmax(category)
