__all__ = ["PseudoHeuristic"]

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=8196)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


class PseudoHeuristic:
    def __init__(self, name):
        print(f"WARNING:{name}:This is a pseudo-heuristic. It is not a real heuristic.")
        self.name = name

    def __call__(self, *args) -> int:
        raise NotImplementedError("This method must be implemented by the subclass.")
