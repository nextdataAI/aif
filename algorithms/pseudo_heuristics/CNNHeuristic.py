__all__ = ["CNNHeuristic"]

import pickle

import keras.optimizers
import wandb
from PIL import Image as im
import os
from data.dataset import ImageDataset
from algorithms.heuristics.Heuristic import Heuristic
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8196)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


class CNNHeuristic(Heuristic):
    def __init__(self, name: str = 'CNN'):
        super().__init__(name=name)
        self.model = self.load_model('data/model.pkl')

    def load_model(self, model_path: str = None):
        """
        Load the model from a path.
        Args:
            model_path (str): The path to the model.
        """
        if model_path is None:
            return None
        try:
            if os.path.exists(model_path):
                self.history = pickle.load(open('data/model_history.pkl', 'rb'))
            if os.path.exists(model_path):
                return pickle.load(open('data/model.pkl', 'rb'))
            # return load_model(model_path)
        except OSError:
            print("The model file does not exist.")
            return None

    def __call__(self, *args) -> float:
        """
        Compute the heuristic value of a state.
        Args:
            state (State): The state to compute the heuristic value.
            _ (State): The goal state.
            game_map (): The game map.
        Returns:
            float: The heuristic value of the state.
        """
        game_map = self.__move_player__(args[0], args[2].get('pixel'), args[3])
        self.__initialize__(game_map)
        return self.model.predict(game_map)

    def __move_player__(self, state, game_map, actions):
        """
        Move the player in a direction.
        Args:
            state (State): The state in which to move the player.
            game_map: The game map.
        Returns:
            new_game_map: The new game map.
        """
        player = np.array(im.open('minihack_images/player.png'), dtype=np.uint8)
        floor = np.array(im.open('minihack_images/floor.png'), dtype=np.uint8)

        def move_player(tmp_image, player_img, floor_img, direction):
            for i in range(512, 800, 16):
                for j in range(288, 976, 16):
                    if np.array_equal(tmp_image[i:i + 16, j:j + 16], player):
                        tmp_image[i:i + 16, j:j + 16] = floor_img
                        if direction == 0:
                            tmp_image[i - 16:i, j:j + 16] = player_img
                        elif direction == 2:
                            tmp_image[i + 16:i + 32, j:j + 16] = player_img
                        elif direction == 3:
                            tmp_image[i:i + 16, j - 16:j] = player_img
                        elif direction == 1:
                            tmp_image[i:i + 16, j + 16:j + 32] = player_img
                        return tmp_image, True
            return tmp_image, False

        # reconstruct the game map from the state and the actions
        actions_to_perform = self.__build_path__(actions, state)

        # move the player in the game map
        for action in actions_to_perform:
            game_map, _ = move_player(game_map, player, floor, action)

        padding = np.zeros((1264, 1264, 3), dtype=np.uint8)
        padding[:, :] = np.array([0, 0, 0])
        padding[464:800, 0:1264] = game_map
        return np.array(padding, dtype=np.float16)

    def __initialize__(self, game_map):
        if self.model is None:
            model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Conv2D(filters=64, strides=(1, 1), kernel_size=(3, 3), padding='same',
                                           activation='relu'),
                    tf.keras.layers.Conv2D(filters=64, strides=(1, 1), kernel_size=(3, 3), padding='same',
                                           activation='relu'),
                    tf.keras.layers.MaxPooling2D((32, 32)),
                    tf.keras.layers.Conv2D(filters=32, strides=(1, 1), kernel_size=(3, 3), padding='same',
                                           activation='relu'),
                    tf.keras.layers.Conv2D(filters=32, strides=(1, 1), kernel_size=(3, 3), padding='same',
                                           activation='relu'),
                    tf.keras.layers.MaxPooling2D((16, 16)),
                    tf.keras.layers.Conv2D(filters=16, strides=(1, 1), kernel_size=(3, 3), padding='same',
                                           activation='relu'),
                    tf.keras.layers.Conv2D(filters=16, strides=(1, 1), kernel_size=(16, 16), padding='same',
                                           activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Dense(512),
                    tf.keras.layers.Dense(32),
                    tf.keras.layers.Dense(1)
                ]
            )
            self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=5,
                                                                   restore_best_weights=True)
            # wandb.init()
            # self.wandb = WandbCallback()
            model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-2, decay=1e-6), loss='mse',
                          run_eagerly=False,
                          metrics=['mae', 'mse'])
            self.model = model

            self.__train__()

    def __train__(self):
        train_dataset = ImageDataset(max_data=1000).__call__()
        val_dataset = ImageDataset(kind='val', max_data=200).__call__()

        # Add model input size based on the dataset
        self.model.build((None, 1264, 1264, 3))

        # train the model
        self.history = self.model.fit(train_dataset, shuffle=True, batch_size=2, callbacks=[self.early_stopping],
                                      epochs=10, validation_data=val_dataset, verbose=1)
        self.model.save('data/model.h5')
        with open('data/model.pkl', 'wb') as file:
            pickle.dump(self.model, file)
        with open('data/model_history.pkl', 'wb') as file:
            pickle.dump(self.history.history, file)

        # evaluate the model
        res = self.model.evaluate(val_dataset, verbose=2)

        # make predictions
        self.model.predict(val_dataset)

        # plot the model
        tf.keras.utils.plot_model(self.model, 'data/model.png', show_shapes=True)

    @staticmethod
    def __build_path__(parent, target):
        path = []
        action = -2
        while target is not None and action != -1:
            path.append(target)
            target, action = parent[target]
        path.reverse()
        return path
