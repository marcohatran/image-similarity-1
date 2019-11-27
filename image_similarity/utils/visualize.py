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


def visualize_tsne(axis: mplt.axes.Axes, x: np.ndarray, labels: np.ndarray):
    if x.ndim != 2:
        raise ValueError(
            "`x` must have two dimensions with its shape equal to (n_samples, n_features)."
        )
    if labels.ndim != 1:
        raise ValueError(
            "`labels` must have one dimensions with its shape equal to (n_samples,)."
        )

    x_embedded = TSNE(n_components=2).fit_transform(x)

    axis.set_aspect("equal")
    # axis.set_ylim(-25, 25)
    # axis.set_xlim(-25, 25)
    # axis.axis("off")
    # axis.axis("tight")
    colors = np.asarray(sns.color_palette("husl", n_colors=10))

    axis.scatter(x_embedded[:, 0], x_embedded[:, 1], lw=0, s=40, c=colors[labels])

    plt.show()
