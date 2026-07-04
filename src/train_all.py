import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from preprocessing import preprocessing_pipeline

from models.lstm import build_lstm
from models.gru import build_gru
from models.cnn_lstm import build_cnn_lstm
from models.attention_lstm import build_attention_lstm


# --------------------------------------------------
# Constants
# --------------------------------------------------

DATA_PATH = os.path.join("..", "data", "raw", "AllCombined_MentalEffortFocus.csv")
SAVED_MODELS_DIR = os.path.join("..", "saved_models")
RESULTS_DIR = os.path.join("..", "results")
METRICS_CSV = os.path.join(RESULTS_DIR, "loso_metrics.csv")

TARGET = "Correct"
WINDOW_SIZE = 30
STRIDE = 5

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
RANDOM_STATE = 42
VAL_SIZE = 0.2

EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 5
MIN_LR = 1e-6

MODEL_BUILDERS = {
    "lstm": build_lstm,
    "gru": build_gru,
    "cnn_lstm": build_cnn_lstm,
    "attention": build_attention_lstm,
}


# --------------------------------------------------
# Setup
# --------------------------------------------------

def ensure_dirs():
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------
# Data splitting / scaling
# --------------------------------------------------

def split_loso_fold(X, y, person_ids, held_out):
    """Split into train (all other participants) and test (held_out)."""

    train_mask = person_ids != held_out
    test_mask = person_ids == held_out

    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


def split_validation(X_train, y_train):
    """Non-participant-wise validation split, stratified on y_train."""

    return train_test_split(
        X_train,
        y_train,
        test_size=VAL_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )


def scale_fold(X_train, X_val, X_test):
    """Fit StandardScaler on training data only, apply to all splits."""

    num_features = X_train.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, num_features))

    def transform(X):
        shape = X.shape
        return scaler.transform(X.reshape(-1, num_features)).reshape(shape)

    return transform(X_train), transform(X_val), transform(X_test)


def get_class_weight(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return dict(zip(classes, weights))


# --------------------------------------------------
# Model building / training
# --------------------------------------------------

def build_and_compile(build_fn, input_shape, num_classes):
    model = build_fn(input_shape, num_classes)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", AUC(name="auc")],
    )

    return model


def get_callbacks(checkpoint_path):
    return [
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_fold(model, X_test, y_test):
    """Evaluate on the held-out participant (not validation)."""

    probs = model.predict(X_test)
    preds = np.argmax(probs, axis=1)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="binary", zero_division=0)
    recall = recall_score(y_test, preds, average="binary", zero_division=0)
    f1 = f1_score(y_test, preds, average="binary", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, probs[:, 1])
    except ValueError:
        roc_auc = np.nan

    return accuracy, precision, recall, f1, roc_auc


# --------------------------------------------------
# Main LOSO loop
# --------------------------------------------------

def run_loso():

    ensure_dirs()

    X, y, person_ids, feature_cols, encoders = preprocessing_pipeline(
        DATA_PATH,
        target=TARGET,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    unique_people = np.unique(person_ids)
    num_classes = len(np.unique(y))
    input_shape = (X.shape[1], X.shape[2])

    fold_records = []

    for model_name, build_fn in MODEL_BUILDERS.items():

        print("\n" + "=" * 60)
        print(f"Model: {model_name}")
        print("=" * 60)

        for held_out in unique_people:

            print(f"\n--- {model_name} | held-out participant: {held_out} ---")

            X_train_full, y_train_full, X_test, y_test = split_loso_fold(
                X, y, person_ids, held_out
            )

            X_train, X_val, y_train, y_val = split_validation(
                X_train_full, y_train_full
            )

            X_train, X_val, X_test = scale_fold(X_train, X_val, X_test)

            # Class weights are computed from the original sparse labels
            # (class_weight expects a {class_index: weight} mapping, which
            # works the same whether targets are sparse or one-hot).
            class_weight = get_class_weight(y_train)

            # One-hot encode for fit/validation only, so shapes match the
            # (batch, num_classes) softmax output required by the AUC
            # metric and categorical_crossentropy. Held-out test labels
            # stay in their original sparse form since sklearn's metrics
            # (used in evaluate_fold) expect sparse class labels.
            y_train_cat = to_categorical(y_train, num_classes=num_classes)
            y_val_cat = to_categorical(y_val, num_classes=num_classes)

            model = build_and_compile(build_fn, input_shape, num_classes)

            checkpoint_path = os.path.join(
                SAVED_MODELS_DIR, f"{model_name}_{held_out}.keras"
            )
            callbacks = get_callbacks(checkpoint_path)

            model.fit(
                X_train,
                y_train_cat,
                validation_data=(X_val, y_val_cat),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1,
            )

            # Reload the best checkpoint saved for this fold and evaluate
            # on the held-out participant (not validation).
            best_model = tf.keras.models.load_model(checkpoint_path)

            accuracy, precision, recall, f1, roc_auc = evaluate_fold(
                best_model, X_test, y_test
            )

            print(
                f"Accuracy={accuracy:.4f}  Precision={precision:.4f}  "
                f"Recall={recall:.4f}  F1={f1:.4f}  ROC-AUC={roc_auc:.4f}"
            )

            fold_records.append({
                "Model": model_name,
                "HeldOutParticipant": held_out,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC_AUC": roc_auc,
            })

    metrics_df = pd.DataFrame(fold_records)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"\nSaved per-fold metrics to {METRICS_CSV}")

    print_summary(metrics_df)


def print_summary(metrics_df):
    print("\n" + "=" * 60)
    print("LOSO Summary (mean +/- std across folds)")
    print("=" * 60)

    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    summary = metrics_df.groupby("Model")[metric_cols].agg(["mean", "std"])

    for model_name in summary.index:
        row = summary.loc[model_name]
        print(f"\n{model_name}")
        for metric in metric_cols:
            print(f"  {metric:<10}: {row[(metric, 'mean')]:.4f} +/- {row[(metric, 'std')]:.4f}")


if __name__ == "__main__":
    run_loso()