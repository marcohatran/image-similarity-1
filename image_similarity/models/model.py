from typing import Tuple, List

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

    def construct_model(
        self, input_shape: List[int], embedding_size: int, learning_rate: float
    ):
        if len(input_shape) != 3:
            raise ValueError("`input_shape` must have three dimensions")

        input_anchor = Input(input_shape, name="input_anchor")
        input_positive = Input(input_shape, name="input_positive")
        input_negative = Input(input_shape, name="input_negative")

        shared_network = Simple.construct_model(input_shape, embedding_size)

        encoded_anchor = shared_network(input_anchor)
        encoded_positive = shared_network(input_positive)
        encoded_negative = shared_network(input_negative)

        merged_embeddings = concatenate(
            [encoded_anchor, encoded_positive, encoded_negative],
            axis=-1,
            name="merged_layer",
        )

        self.triplet_model = Model(
            inputs=[input_anchor, input_positive, input_negative],
            outputs=merged_embeddings,
            name="triplet_model",
        )
        self.triplet_model.compile(
            optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999),
            loss=triplet_loss,
        )

        self.embedding_model = Model(
            inputs=input_anchor, outputs=encoded_anchor, name="embedding_model"
        )

    def summary(self):
        print(self.triplet_model.summary())
        print()
        print(self.embedding_model.summary())

    def fit(
        self,
        X: np.ndarray,
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
        self.history = self.triplet_model.fit(
            X,
            np.empty((X[0].shape[0], embedding_size * 3)),
            epochs=epochs,
            shuffle=shuffle,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split,
            validation_data=validation_data,
            # callbacks=callbacks,
        )

    def predict(self, X: np.ndarray):
        return self.model.predict(X)

    def evaluate(self, test_set, verbose=1):
        raise NotImplementedError

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


def triplet_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha=0.4):
    print("y_pred.shape = ", y_pred.shape)
    total_length = y_pred.shape.as_list()[-1]

    anchor = y_pred[:, 0 : int(total_length * 1 / 3)]
    positive = y_pred[:, int(total_length * 1 / 3) : int(total_length * 2 / 3)]
    negative = y_pred[:, int(total_length * 2 / 3) : int(total_length * 3 / 3)]

    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    loss = K.maximum(pos_dist - neg_dist + alpha, 0.0)
    return loss
