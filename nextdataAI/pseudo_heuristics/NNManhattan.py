__all__ = ["NNManhattan"]
import sys
from pathlib import Path
import os

import keras.backend as K
import numpy as np
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

from data import AsciiDataset
from .PseudoHeuristics import PseudoHeuristic

sys.path.append(str(Path.cwd().parent))


def mee(y_true, y_pred):
    """
    mee
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


class NNManhattan(PseudoHeuristic):
    def __init__(self, name: str = "NNHeuristic", model='NNManhattan'):
        super().__init__(name=name)
        self.model_path = model
        try:
            self.model = tf.keras.models.load_model(custom_objects={'mee': mee},
                                                    filepath=f"{os.path.dirname(__file__)}/models/{self.model_path}.keras")
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
        target = args[1]
        start = args[0]
        x1, y1 = start
        x2, y2 = target
        return self.__predict__(np.array([x1, y1, x2, y2]).reshape(1, -1))

    def __train__(self):
        train_data, train_labels = AsciiDataset(kind='train').__call_2__()
        val_data, val_labels = AsciiDataset(kind='val').__call_2__()
        self.input_shape = train_data[0].shape

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(200, activation='sigmoid'),
            tf.keras.layers.Dense(200, activation='sigmoid'),
            tf.keras.layers.Dense(512, activation='sigmoid'),
            tf.keras.layers.Dense(128, activation='linear'),
            tf.keras.layers.Dense(1)
        ])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_mse',
            verbose=1,
            min_delta=1e-4,
            patience=20,
            mode='min',
            restore_best_weights=True)

        wandb.init()
        wandb_callback = WandbMetricsLogger(log_freq='epoch')

        model.compile(
            optimizer=tf.keras.optimizers.SGD(weight_decay=1e-8, learning_rate=5e-5, momentum=0.6, nesterov=False),
            loss='mse',
            metrics=['mse', 'mae', mee])

        # compose the model
        model.build((None, 4))
        model.summary()

        # train the model
        model.fit(x=train_data, y=train_labels, epochs=100, batch_size=128, callbacks=[early_stopping, wandb_callback],
                  validation_data=(val_data, val_labels), verbose=1)

        # evaluate the model
        test_data, test_labels = AsciiDataset(kind='test').__call_2__()
        scores = model.evaluate(test_data, test_labels, verbose=0)
        print(scores)

        # save the model
        model.save(f"{os.path.dirname(__file__)}/models/{self.model_path}.keras")
        self.model = model

    def __predict__(self, state):
        out = self.model.predict(state, verbose=0)[0][0]
        return out
