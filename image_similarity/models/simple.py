from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Lambda,
    Dense,
    Flatten,
    MaxPooling2D,
    concatenate,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.losses import binary_crossentropy


class Simple:
    def __init__(self):
        raise NotImplementedError()

    def __call__(self, inputs):
        raise NotImplementedError()

    @classmethod
    def construct_model(self, input_shape):
        net = Conv2D(
            128,
            (7, 7),
            input_shape=input_shape,
            padding="same",
            activation="relu",
            name="conv1",
        )
        net = MaxPooling2D((2, 2), (2, 2), padding="same", name="pool1")(net)
        net = Conv2D(256, (5, 5), padding="same", activation="relu", name="conv2")
        net = MaxPooling2D((2, 2), (2, 2), padding="same", name="pool2")
        net = Flatten(name="flatten")
        net = Dense(4, name="embeddings")
        return net

