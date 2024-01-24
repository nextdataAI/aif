__all__ = ["NN"]

import pickle
import os
import keras.backend as K
import numpy as np
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

from data import AsciiDataset
from .PseudoHeuristics import PseudoHeuristic


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


class NN(PseudoHeuristic):
    def __init__(self, name: str = "NNHeuristic"):
        super().__init__(name=name)
        try:
            self.model = tf.keras.models.load_model(filepath=f"{os.path.dirname(__file__)}/models/NNHeuristic")
            with open(f'{os.path.dirname(__file__)}/models/NNHeuristicmax.pkl', 'rb') as file:
                self.max = pickle.load(file)
        except Exception as e:
            # Print exception message
            print(e)

            self.model = None
            self.__train__()

    def __call__(self, *args):
        """
        :type args: many types
        :param args: start, target, others
        :return: estimation of the distance between start and target
        """
        self.game_map = args[2].get('chars')
        self.game_map = (np.array(self.game_map).ravel()) / 255
        return self.__predict__(self.game_map)

    def __train__(self):
        train_data, train_labels = AsciiDataset(kind='train').__call__()
        val_data, val_labels = AsciiDataset(kind='val').__call__()
        self.input_shape = train_data[0].shape
        self.max = train_labels.max() if train_labels.max() > val_labels.max() else val_labels.max()
        train_labels = 1 - (np.array(train_labels) / train_labels.max())
        val_labels = 1 - (np.array(val_labels) / val_labels.max())

        model = tf.keras.Sequential([
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            # tf.keras.layers.Dense(256, activation='sigmoid'),
            tf.keras.layers.Dense(128, activation='sigmoid'),
            tf.keras.layers.Dense(1)
        ])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_mse',
            verbose=1,
            min_delta=1e-2,
            patience=10,
            mode='min',
            restore_best_weights=True)

        wandb.init()
        wandb_callback = WandbMetricsLogger(log_freq='epoch')

        model.compile(
            optimizer=tf.keras.optimizers.SGD(weight_decay=1e-8, learning_rate=1e-4, momentum=0.7, nesterov=False),
            loss='mse',
            metrics=['mse', 'mae'])

        # compose the model
        model.build((None, 1659))
        model.summary()

        # train the model
        model.fit(x=train_data, y=train_labels, epochs=100, batch_size=128, callbacks=[early_stopping, wandb_callback],
                  validation_data=(val_data, val_labels), verbose=1)

        # save the model
        model.save(f"{os.path.dirname(__file__)}/models/NNHeuristic")
        self.model = model
        with open(f'{os.path.dirname(__file__)}/models/NNHeuristicmax.pkl', 'wb') as file:
            pickle.dump(self.max, file)

    def __predict__(self, state):
        out = int(1 - self.model.predict(state.reshape(1, -1) / 255, verbose=0)[0][0] * self.max)

        return out
