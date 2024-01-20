__all__ = ["CNNHeuristic"]

from PIL import Image as im
from data.image_dataset import ImageDataset
from algorithms.heuristics.Heuristic import Heuristic
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np


class CNNHeuristic(Heuristic):
    def __init__(self, name: str = 'CNN'):
        super().__init__(name=name)
        self.model = self.load_model('data/model.h5')

    def load_model(self, model_path: str = None):
        """
        Load the model from a path.
        Args:
            model_path (str): The path to the model.
        """
        if model_path is None:
            return None
        try:
            return load_model(model_path)
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
        game_map = self.__move_player__(args[0], args[2], args[3])
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
                    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(1)
                ]
            )
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
            # wandb = WandbCallback()
            model.compile(optimizer='adam', loss='mse', run_eagerly=False,
                          metrics=['mae', 'mse', 'accuracy'])
            self.model = model
            self.__train__()

    def __train__(self):
        data = []
        labels = []

        dataset = ImageDataset(max_data=10000)
        data, labels = dataset()

        data = np.array(data, dtype=np.float16)
        labels = np.array(labels, dtype=np.float16)

        # split the data into train and validation
        train_data = data[:int(len(data) * 0.8)]
        train_labels = labels[:int(len(labels) * 0.8)]
        val_data = data[int(len(data) * 0.8):]
        val_labels = labels[int(len(labels) * 0.8):]

        # train the model
        self.model.fit(train_data, train_labels, shuffle=True, batch_size=2, epochs=10, validation_data=(val_data, val_labels))
        self.model.save('data/model.h5')

        # evaluate the model
        self.model.evaluate(val_data, val_labels, verbose=2)

        # make predictions
        self.model.predict(val_data)

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
