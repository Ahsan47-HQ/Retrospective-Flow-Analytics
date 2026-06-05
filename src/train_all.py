import os
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from models.lstm import build_lstm
from models.gru import build_gru
from models.cnn_lstm import build_cnn_lstm
from models.attention_lstm import build_attention_lstm


# -----------------------
# Config
# -----------------------

EPOCHS = 50
BATCH = 16


os.makedirs(
    "../saved_models",
    exist_ok=True
)


# -----------------------
# Load data
# -----------------------

X_train = np.load("../data/processed/X_train.npy")
X_val = np.load("../data/processed/X_val.npy")

y_train = np.load("../data/processed/y_train.npy")
y_val = np.load("../data/processed/y_val.npy")



input_shape = (
    X_train.shape[1],
    X_train.shape[2]
)

num_classes = len(
    np.unique(y_train)
)



# -----------------------
# Models
# -----------------------

models = {

    "lstm":
        build_lstm,

    "gru":
        build_gru,

    "cnn_lstm":
        build_cnn_lstm,

    "attention":
        build_attention_lstm
}



# class imbalance

weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(y_train),

    y=y_train

)

class_weights = dict(
    zip(
        np.unique(y_train),
        weights
    )
)



# -----------------------
# Train loop
# -----------------------

for name, builder in models.items():


    print(
        "\n======================"
    )

    print(
        "Training:",
        name
    )


    model = builder(
        input_shape,
        num_classes
    )



    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )



    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=10,

            restore_best_weights=True

        ),

        tf.keras.callbacks.ModelCheckpoint(

            f"../saved_models/{name}.keras",

            save_best_only=True,

            monitor="val_loss"

        )

    ]



    model.fit(

        X_train,

        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1

    )


print("All training complete")