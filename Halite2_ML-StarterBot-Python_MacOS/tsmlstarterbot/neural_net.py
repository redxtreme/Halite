import os
import tensorflow as tf
import numpy as np

from tsmlstarterbot.common import PLANET_MAX_NUM, PER_PLANET_FEATURES

# We don't want tensorflow to produce any warnings in the standard output, since the bot communicates
# with the game engine through stdout/stdin.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'
tf.logging.set_verbosity(tf.logging.ERROR)


# Normalize planet features within each frame.
def normalize_input(input_data):

    # Assert the shape is what we expect
    shape = input_data.shape
    assert len(shape) == 3 and shape[1] == PLANET_MAX_NUM and shape[2] == PER_PLANET_FEATURES

    m = np.expand_dims(input_data.mean(axis=1), axis=1)
    s = np.expand_dims(input_data.std(axis=1), axis=1)
    return (input_data - m) / (s + 1e-6)


class NeuralNet(object):
    FIRST_LAYER_SIZE = 12
    SECOND_LAYER_SIZE = 20
    THIRD_LAYER_SIZE = 12
    FOURTH_LAYER_SIZE = 12
    FIFTH_LAYER_SIZE = 20
    SIXTH_LAYER_SIZE = 12
    SEVENTH_LAYER_SIZE = 10
    EIGTH_LAYER_SIZE = 20
    NINETH_LAYER_SIZE = 12
    TENTH_LAYER_SIZE = 20
    ELEVENTH_LAYER_SIZE = 12
    TWELVTH_LAYER_SIZE = 12

    def __init__(self, cached_model=None, seed=None):
        self._graph = tf.Graph()

        with self._graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self._session = tf.Session()
            self._features = tf.placeholder(dtype=tf.float32, name="input_features",
                                            shape=(None, PLANET_MAX_NUM, PER_PLANET_FEATURES))

            # target_distribution describes what the bot did in a real game.
            # For instance, if it sent 20% of the ships to the first planet and 15% of the ships to the second planet,
            # then expected_distribution = [0.2, 0.15 ...]
            self._target_distribution = tf.placeholder(dtype=tf.float32, name="target_distribution",
                                                       shape=(None, PLANET_MAX_NUM))

            # Combine all the planets from all the frames together, so it's easier to share
            # the weights and biases between them in the network.
            flattened_frames = tf.reshape(self._features, [-1, PER_PLANET_FEATURES])

            first_layer = tf.contrib.layers.fully_connected(flattened_frames, self.FIRST_LAYER_SIZE)
            second_layer = tf.contrib.layers.fully_connected(first_layer, self.SECOND_LAYER_SIZE)
            third_layer = tf.contrib.layers.fully_connected(second_layer, self.THIRD_LAYER_SIZE)
            fourth_layer = tf.contrib.layers.fully_connected(third_layer, self.FOURTH_LAYER_SIZE)
            fifth_layer = tf.contrib.layers.fully_connected(fourth_layer, self.FIFTH_LAYER_SIZE)
            sixth_layer = tf.contrib.layers.fully_connected(fifth_layer, self.SIXTH_LAYER_SIZE)
            seventh_layer = tf.contrib.layers.fully_connected(flattened_frames, self.SEVENTH_LAYER_SIZE)
            eigth_layer = tf.contrib.layers.fully_connected(first_layer, self.EIGTH_LAYER_SIZE)
            nineth_layer = tf.contrib.layers.fully_connected(second_layer, self.NINETH_LAYER_SIZE)
            tenth_layer = tf.contrib.layers.fully_connected(third_layer, self.TENTH_LAYER_SIZE)
            eleventh_layer = tf.contrib.layers.fully_connected(fourth_layer, self.ELEVENTH_LAYER_SIZE)
            twevlth_layer = tf.contrib.layers.fully_connected(fifth_layer, self.TWELVTH_LAYER_SIZE)

            layer_13 = tf.contrib.layers.fully_connected(twevlth_layer, self.FIRST_LAYER_SIZE)
            layer_14 = tf.contrib.layers.fully_connected(layer_13, self.SECOND_LAYER_SIZE)
            layer_15 = tf.contrib.layers.fully_connected(layer_14, self.THIRD_LAYER_SIZE)
            layer_16 = tf.contrib.layers.fully_connected(layer_15, self.FOURTH_LAYER_SIZE)
            layer_17 = tf.contrib.layers.fully_connected(layer_16, self.FIFTH_LAYER_SIZE)
            layer_18 = tf.contrib.layers.fully_connected(layer_17, self.SIXTH_LAYER_SIZE)
            layer_19 = tf.contrib.layers.fully_connected(layer_18, self.SEVENTH_LAYER_SIZE)
            layer_20 = tf.contrib.layers.fully_connected(layer_19, self.EIGTH_LAYER_SIZE)
            layer_21 = tf.contrib.layers.fully_connected(layer_20, self.NINETH_LAYER_SIZE)
            layer_22 = tf.contrib.layers.fully_connected(layer_21, self.TENTH_LAYER_SIZE)
            layer_23 = tf.contrib.layers.fully_connected(layer_22, self.ELEVENTH_LAYER_SIZE)
            layer_24 = tf.contrib.layers.fully_connected(layer_23, self.TWELVTH_LAYER_SIZE)

            layer_25 = tf.contrib.layers.fully_connected(layer_24, self.FIRST_LAYER_SIZE)
            layer_26 = tf.contrib.layers.fully_connected(layer_25, self.SECOND_LAYER_SIZE)
            layer_27 = tf.contrib.layers.fully_connected(layer_26, self.THIRD_LAYER_SIZE)
            layer_28 = tf.contrib.layers.fully_connected(layer_27, self.FOURTH_LAYER_SIZE)
            layer_29 = tf.contrib.layers.fully_connected(layer_28, self.FIFTH_LAYER_SIZE)
            layer_30 = tf.contrib.layers.fully_connected(layer_29, self.SIXTH_LAYER_SIZE)
            layer_31 = tf.contrib.layers.fully_connected(layer_30, self.SEVENTH_LAYER_SIZE)
            layer_32 = tf.contrib.layers.fully_connected(layer_31, self.EIGTH_LAYER_SIZE)
            layer_33 = tf.contrib.layers.fully_connected(layer_32, self.NINETH_LAYER_SIZE)
            layer_34 = tf.contrib.layers.fully_connected(layer_33, self.TENTH_LAYER_SIZE)
            layer_35 = tf.contrib.layers.fully_connected(layer_34, self.ELEVENTH_LAYER_SIZE)
            layer_36 = tf.contrib.layers.fully_connected(layer_35, self.TWELVTH_LAYER_SIZE)

            final_layer = tf.contrib.layers.fully_connected(layer_24, 1, activation_fn=None)

            # Group the planets back in frames.
            logits = tf.reshape(final_layer, [-1, PLANET_MAX_NUM])

            self._prediction_normalized = tf.nn.softmax(logits)

            self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self._target_distribution))

            self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self._loss)
            self._saver = tf.train.Saver()

            if cached_model is None:
                self._session.run(tf.global_variables_initializer())
            else:
                self._saver.restore(self._session, cached_model)

    def fit(self, input_data, expected_output_data):
        """
        Perform one step of training on the training data.

        :param input_data: numpy array of shape (number of frames, PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        loss, _ = self._session.run([self._loss, self._optimizer],
                                    feed_dict={self._features: normalize_input(input_data),
                                               self._target_distribution: expected_output_data})
        return loss

    def predict(self, input_data):
        """
        Given data from 1 frame, predict where the ships should be sent.

        :param input_data: numpy array of shape (PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :return: 1-D numpy array of length (PLANET_MAX_NUM) describing percentage of ships
        that should be sent to each planet
        """
        return self._session.run(self._prediction_normalized,
                                 feed_dict={self._features: normalize_input(np.array([input_data]))})[0]

    def compute_loss(self, input_data, expected_output_data):
        """
        Compute loss on the input data without running any training.

        :param input_data: numpy array of shape (number of frames, PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        return self._session.run(self._loss,
                                 feed_dict={self._features: normalize_input(input_data),
                                            self._target_distribution: expected_output_data})

    def save(self, path):
        """
        Serializes this neural net to given path.
        :param path:
        """
        self._saver.save(self._session, path)
