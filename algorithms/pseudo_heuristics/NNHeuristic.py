__all__ = ["NNHeuristic"]

from exclusiveAI.Composer import Composer
import numpy as np
from exclusiveAI import *
from algorithms.heuristics import Heuristic
from data import AsciiDataset


class NNHeuristic(Heuristic):
    def __init__(self, name: str = "NNHeuristic"):
        super().__init__(name=name)
        try:
            self.model = NeuralNetwork.NeuralNetwork.load("algorithms/pseudo_heuristics/models/nn_model.pkl")
        except:
            self.model = None
            self.__train__()

    def __call__(self, *args):
        """
        :type args: many types
        :param args: start, target, others
        :return: estimation of the distance between start and target
        """
        self.game_map = args[2].get('chars')
        self.game_map = np.array(self.game_map).ravel()
        self.input_shape = self.game_map.shape
        self.output_shape = (1,)
        return 0

    def __train__(self):
        train_data, train_labels = AsciiDataset(kind='train').__call__()
        val_data, val_labels = AsciiDataset(kind='val').__call__()
        self.input_shape = train_data[0].shape
        model: NeuralNetwork.NeuralNetwork = Composer(
            input_shape=self.input_shape,
            regularization=0.0001,
            learning_rate=0.001,
            loss_function="mse",
            optimizer="sgd",
            activation_functions=["relu", "sigmoid"],
            output_activation='linear',
            num_of_units=[128, 32],
            num_layers=2,
            model_name='NNHeuristic',
            nesterov=True,
            momentum=0.9,
            initializers=['gaussian'],
            callbacks=['early_stopping_1e-2_10_True', 'wandb'],
            verbose=True,
            outputs=1,
        ).compose(regression=True)

        # train the model
        model.train(inputs=self.game_map, input_label=train_labels, epochs=1000, batch_size=32,
                    val=val_data, val_labels=val_labels)

        # save the model
        model.save("algorithms/pseudo_heuristics/models/nn_model.pkl")
        self.model = model
