"""
Baselines.py
============
Deep baselines used in the intra-subject exploration, reported as Conv, Dense,
and Recur so that generic architecture names are not foregrounded in results.
These complement the proposed model in ModelArchitecture.py.
"""
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     Dense, LSTM, Reshape, concatenate)
from tensorflow.keras.models import Model


def _inputs(eeg_shape, nirs_shape):
    return (Input(shape=eeg_shape, name="eeg_input"),
            Input(shape=nirs_shape, name="nirs_input"))


def build_conv(eeg_shape, nirs_shape, num_classes):
    def stream(inp):
        x = Conv2D(32, (3, 3), activation="relu")(inp)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation="relu")(x)
        return Flatten()(MaxPooling2D((2, 2))(x))
    e, n = _inputs(eeg_shape, nirs_shape)
    h = Dense(64, activation="relu")(concatenate([stream(e), stream(n)]))
    return Model([e, n], Dense(num_classes, activation="softmax")(h), name="Conv")


def build_dense(eeg_shape, nirs_shape, num_classes):
    e, n = _inputs(eeg_shape, nirs_shape)
    h = concatenate([Flatten()(e), Flatten()(n)])
    for _ in range(3):
        h = Dense(128, activation="relu")(h)
    return Model([e, n], Dense(num_classes, activation="softmax")(h), name="Dense")


def build_recur(eeg_shape, nirs_shape, num_classes):
    def stream(inp):
        return LSTM(64)(Reshape((inp.shape[1], -1))(inp))
    e, n = _inputs(eeg_shape, nirs_shape)
    h = Dense(64, activation="relu")(concatenate([stream(e), stream(n)]))
    return Model([e, n], Dense(num_classes, activation="softmax")(h), name="Recur")


BASELINES = {"Conv": build_conv, "Dense": build_dense, "Recur": build_recur}
