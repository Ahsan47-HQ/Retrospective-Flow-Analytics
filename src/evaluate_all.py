import os
import csv

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)



os.makedirs(
    "../results",
    exist_ok=True
)



X_test = np.load(
    "../data/processed/X_test.npy"
)

y_test = np.load(
    "../data/processed/y_test.npy"
)



models = [

    "lstm",

    "gru",

    "cnn_lstm",

    "attention"

]



results = []



for name in models:


    print(
        "Evaluating:",
        name
    )


    model = tf.keras.models.load_model(

        f"../saved_models/{name}.keras"

    )



    probs = model.predict(
        X_test
    )


    preds = np.argmax(
        probs,
        axis=1
    )



    acc = accuracy_score(
        y_test,
        preds
    )

    precision = precision_score(
        y_test,
        preds
    )

    recall = recall_score(
        y_test,
        preds
    )

    f1 = f1_score(
        y_test,
        preds
    )



    results.append(
        [
            name,
            acc,
            precision,
            recall,
            f1
        ]
    )



    # confusion matrix

    cm = confusion_matrix(
        y_test,
        preds
    )


    plt.figure(
        figsize=(5,4)
    )

    plt.imshow(cm)

    plt.title(
        name
    )


    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                cm[i,j],
                ha="center",
                va="center"
            )


    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )


    plt.savefig(
        f"../results/{name}_confusion.png"
    )


    plt.close()



# save csv


with open(
    "../results/metrics.csv",
    "w",
    newline=""
) as f:


    writer = csv.writer(f)


    writer.writerow(

        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]

    )


    writer.writerows(
        results
    )



print("\nSaved results!")