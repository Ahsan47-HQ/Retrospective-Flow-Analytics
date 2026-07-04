from keras.models import Sequential

from keras.layers import (
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout
)



def build_cnn_lstm(
    input_shape,
    num_classes
):

    model = Sequential([

        Conv1D(

            filters=64,

            kernel_size=3,

            activation="relu",

            input_shape=input_shape

        ),

        MaxPooling1D(
            pool_size=2
        ),

        LSTM(
            32,
            return_sequences=True
        ),

        Dropout(
            0.35
        ),

        LSTM(
            32
        ),

        Dropout(
            0.25
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