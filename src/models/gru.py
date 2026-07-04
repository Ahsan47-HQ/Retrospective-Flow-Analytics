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
            64,
            input_shape=input_shape,
            return_sequences=True
        ),


        Dropout(
            0.3
        ),

        GRU(
            32
        ),


        Dropout(
            0.3
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