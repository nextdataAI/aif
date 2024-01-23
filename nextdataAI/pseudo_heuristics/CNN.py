__all__ = ["CNN"]
import os
import numpy as np
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

from data.dataset import ImageDataset
from data.dataset import get_player_floor_target_png
from .PseudoHeuristics import PseudoHeuristic


class CNN(PseudoHeuristic):
    def __init__(self, name: str = 'CNN'):
        super().__init__(name=name)
        try:
            self.model = tf.keras.models.load_model(filepath=f"{os.path.dirname(__file__)}/models/CNNHeuristic")
        except Exception as e:
            # Print exception message
            print(e)

            self.model = None
            self.__train__()

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
        return self.__predict__(game_map)

    def __move_player__(self, state, game_map, actions):
        """
        Move the player in a direction.
        Args:
            state (State): The state in which to move the player.
            game_map: The game map.
        Returns:
            new_game_map: The new game map.
        """
        player, floor, _ = get_player_floor_target_png()

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

    def __train__(self):
        train_data, train_labels = ImageDataset(kind='train')
        val_data, val_labels = ImageDataset(kind='val')
        self.input_shape = train_data[0].shape

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
            metrics=['mse', 'mae'])

        # compose the model
        model.build((None, 4))
        model.summary()

        # train the model
        model.fit(x=train_data, y=train_labels, epochs=100, batch_size=128, callbacks=[early_stopping, wandb_callback],
                  validation_data=(val_data, val_labels), verbose=1)

        # evaluate the model
        test_data, test_labels = ImageDataset(kind='test')
        scores = model.evaluate(test_data, test_labels, verbose=0)
        print(scores)

        # save the model
        model.save(f"{os.path.dirname(__file__)}/models/CNNHeuristic")
        self.model = model

    def __predict__(self, state):
        out = self.model.predict(state, verbose=0)[0][0]
        return out

    @staticmethod
    def __build_path__(parent, target):
        path = []
        action = -2
        while target is not None and action != -1:
            path.append(target)
            target, action = parent[target]
        path.reverse()
        return path
