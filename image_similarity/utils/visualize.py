import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras as tfk
from tensorflow.keras.datasets import mnist
from sklearn.manifold import TSNE

friendly_colors = {
    "orange": "#E69F00",
    "blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "red": "#D55E00",
    "darkblue": "#0072B2",
    "pink": "#CC79A7",
}


def visualize_tsne_v1(
    axis: mplt.axes.Axes, x: np.ndarray, labels: np.ndarray, n_samples: int
):
    if x.ndim != 4:
        raise ValueError("`x` must have three dimensions.")
    if labels.ndim != 1:
        raise ValueError(
            "`labels` must have one dimensions with its shape equal to (n_samples,)."
        )

    x_flatten = x.reshape(x.shape[0], -1)[:n_samples, :]
    x_embedded = TSNE(n_components=2).fit_transform(x_flatten)
    labels_selected = labels[:n_samples]

    axis.set_aspect("equal")
    # axis.set_ylim(-25, 25)
    # axis.set_xlim(-25, 25)
    # axis.axis("off")
    # axis.axis("tight")
    colors = np.asarray(sns.color_palette("husl", n_colors=10))

    axis.scatter(
        x_embedded[:, 0], x_embedded[:, 1], lw=0, s=40, c=colors[labels_selected]
    )


def visualize_tsne_v2(
    axis: mplt.axes.Axes, x: np.ndarray, labels: np.ndarray, n_samples: int
):
    if x.ndim != 4:
        raise ValueError("`x` must have three dimensions.")
    if labels.ndim != 1:
        raise ValueError(
            "`labels` must have one dimensions with its shape equal to (n_samples,)."
        )

    x_flatten = x.reshape(x.shape[0], -1)[:n_samples, :]
    x_embedded = TSNE(n_components=2).fit_transform(x_flatten)
    labels_selected = labels[:n_samples]
    unique_labels = np.unique(labels_selected)
    print("Unique labels:", unique_labels)

    axis.set_aspect("equal")
    colors = np.asarray(sns.color_palette("husl", n_colors=unique_labels.shape[0]))
    for idx, label in enumerate(unique_labels):
        axis.plot(
            x_embedded[labels_selected == label, 0],
            x_embedded[labels_selected == label, 1],
            ".",
            color=colors[idx],
            alpha=0.8,
            label=label,
        )
    axis.legend()
