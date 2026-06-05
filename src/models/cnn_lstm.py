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

            filters=32,

            kernel_size=3,

            activation="relu",

            input_shape=input_shape

        ),



        MaxPooling1D(
            pool_size=2
        ),



        LSTM(
            32
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