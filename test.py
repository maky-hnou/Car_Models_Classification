import cv2
import tensorflow as tf
from resnet50 import ResNet


class Test:
    def __init__(self, image, graph_path):
        self.image = image
        self.graph_path = graph_path

    def predict(self):
        resnet = ResNet()
        Y_hat, model_params = resnet.build_model()
        X = model_params['input']
        saver = tf.train.Saver()

        with tf.Session() as session:
            saver.restore(session, self.graph_path)
            category = session.run(Y_hat, feed_dict={X: [self.image]})
