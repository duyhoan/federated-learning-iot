# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras


class SimpleCNN:
    """
    Convolution Neural Network for Fashion-MNIST classification
    """

    def __init__(self, image, label, learning_rate, global_step, n_output_classes):
        self._image = image
        self._label = label
        self._n_output_classes = n_output_classes
        # Object to keep moving averages of our metrics (for tensorboard)
        self.summary_averages = tf.train.ExponentialMovingAverage(0.9)
        self._model_definition(learning_rate, global_step)

    def _conv_net(self):
        """
        Defining model layers
        :return: final output layer
        """
        flatten_layer = tf.layers.flatten(self._image, name='flatten')

        dense_layer = tf.layers.dense(flatten_layer, 128, activation=tf.nn.relu, name='relu')

        # Output layer, class prediction
        output = tf.layers.dense(dense_layer, self._n_output_classes, activation=tf.nn.softmax, name='softmax')

        return output

    def _model_definition(self, learning_rate, global_step):
        # Get the final layer from pre-defined cnn
        output = self._conv_net()

        # Define cross_entropy loss
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(self._label, output))
            self.loss_averages_op = self.summary_averages.apply([self.loss])
            # Store moving average of the loss
            tf.summary.scalar('cross_entropy', self.summary_averages.average(self.loss))

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                # Compare prediction with actual label
                correct_prediction = tf.equal(tf.argmax(output, 1),
                                              tf.cast(self._label, tf.int64))
            # Average correct predictions in the current batch
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_metric')
            self.accuracy_averages_op = self.summary_averages.apply([self.accuracy])
            # Store moving average of the accuracy
            tf.summary.scalar('accuracy', self.summary_averages.average(self.accuracy))

        # Define optimizer and training op
        with tf.name_scope('train'):
            # Make train_op dependent on moving averages ops. Otherwise they will be
            # disconnected from the graph
            with tf.control_dependencies([self.loss_averages_op, self.accuracy_averages_op]):
                self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)
