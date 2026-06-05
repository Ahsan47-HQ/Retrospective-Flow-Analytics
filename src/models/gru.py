from keras.models import Sequential
from keras.layers import (
    GRU,
    Dense,
    Dropout
)


def build_gru(
    input_shape,
    num_classes
):

    model = Sequential([

        GRU(
            32,
            input_shape=input_shape
        ),


        Dropout(
            0.4
        ),


        Dense(
            16,
            activation="relu"
        ),


        Dense(
            num_classes,
            activation="softmax"
        )

    ])

    return model