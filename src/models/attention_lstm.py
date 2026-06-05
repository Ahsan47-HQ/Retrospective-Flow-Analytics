import tensorflow as tf

from keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D
)

from keras.models import Model



def build_attention_lstm(
    input_shape,
    num_classes
):


    inputs = Input(
        shape=input_shape
    )



    x = LSTM(

        32,

        return_sequences=True

    )(inputs)



    attention = Attention()(

        [
            x,
            x
        ]

    )



    x = GlobalAveragePooling1D()(

        attention

    )



    x = Dropout(
        0.4
    )(x)



    x = Dense(

        16,

        activation="relu"

    )(x)



    outputs = Dense(

        num_classes,

        activation="softmax"

    )(x)



    model = Model(

        inputs,

        outputs

    )


    return model