from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm(input_shape, num_classes):

    model = Sequential([

        LSTM(
            64,
            input_shape=input_shape,
            return_sequences=True
        ),

        Dropout(0.3),

        LSTM(
            32
        ),

        Dropout(0.3),

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