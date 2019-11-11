from typing import Tuple

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model

from .simple import Simple
from ..utils import GetBest


class ImageSimilarityModel:
    def __init__(self):
        self.model = None
        self.history = None

    def construct_model(self, learning_rate: float):
        input_anchor = Input([28, 28, 1], name="input_anchor")
        input_positive = Input([28, 28, 1], name="input_positive")
        input_negative = Input([28, 28, 1], name="input_negative")

        shared_network = Simple.construct_model([28, 28, 1, 1])

        encoded_anchor = shared_network(input_anchor)
        encoded_positive = shared_network(input_positive)
        encoded_negative = shared_network(input_negative)

        merged_vector = concatenate(
            [encoded_anchor, encoded_positive, encoded_negative],
            axis=-1,
            name="merged_layer",
        )

        self.triplet_model = Model(
            inputs=[input_anchor, input_positive, input_negative], outputs=merged_vector
        )
        self.triplet_model.compile(
            optimizer=Adam(lr=learning_rate, clipnorm=1.0), loss=triplet_loss
        )

        self.embedding_model = Model(inputs=input_anchor, outputs=encoded_anchor)

    def summary(self):
        print(self.model.summary())

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        shuffle: bool,
        batch_size: int,
        validation_split: int = None,
        validation_data: Tuple[np.ndarray, np.ndarray] = None,
        verbose: int = 2,
    ):
        callbacks = [
            GetBest(
                monitor="loss"
                if validation_data is None and validation_split is None
                else "val_loss",
                verbose=verbose,
                mode="min",
            )
        ]
        self.history = self.model.fit(
            X,
            y,
            epochs=epochs,
            shuffle=shuffle,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
        )

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, test_set, verbose=1):
        pass

    def visualize_learning_curves(self, axis):
        axis.set_title("Learning curves")
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.plot(self.history.history["loss"], label="train")
        if "val_loss" in self.history.history:
            axis.plot(self.history.history["val_loss"], label="validation")
        axis.legend(loc="upper right")

    def save(self, model_path):
        self.model.save(model_path)

    def save_weights(self, model_path):
        self.model.save_weights(model_path)

    def load(self, model_path):
        self.model = load_model(model_path)

    def load_weights(self, model_path):
        self.model.load_weights(model_path)


def triplet_loss(y_true, y_pred, alpha=0.4):
    print("y_pred.shape = ", y_pred.shape)
    total_length = y_pred.shape.as_list()[-1]

    anchor = y_pred[:, 0 : int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3) : int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3) : int(total_lenght * 3 / 3)]

    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    loss = K.maximum(post_dist - neg_dist + alpha, 0.0)
    return loss
