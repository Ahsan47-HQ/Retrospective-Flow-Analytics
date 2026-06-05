from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm(input_shape, num_classes):

    model = Sequential([

        LSTM(
            16,
            input_shape=input_shape
        ),

        Dropout(0.4),

        Dense(
            8,
            activation="relu"
        ),

        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model