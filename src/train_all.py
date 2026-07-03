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
    confusion_matrix,
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocessing import preprocessing_pipeline

from models.lstm import build_lstm
from models.gru import build_gru
from models.cnn_lstm import build_cnn_lstm
from models.attention_lstm import build_attention_lstm


# --------------------------------------------------
# Config
# --------------------------------------------------

EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001

DATA_PATH = "../data/raw/AllCombined_MentalEffortFocus.csv"
SAVED_MODELS_DIR = "../saved_models"
RESULTS_DIR = "../results"
METRICS_CSV = os.path.join(RESULTS_DIR, "loso_metrics.csv")

MODEL_BUILDERS = {
    "lstm": build_lstm,
    "gru": build_gru,
    "cnn_lstm": build_cnn_lstm,
    "attention": build_attention_lstm,
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def ensure_dirs():
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def scale_fold(X_train, X_val, X_test):
    """Fit a StandardScaler on training data only, apply to all splits."""

    num_features = X_train.shape[-1]

    scaler = StandardScaler()

    X_train_flat = X_train.reshape(-1, num_features)
    scaler.fit(X_train_flat)

    def transform(X):
        shape = X.shape
        X_flat = X.reshape(-1, num_features)
        X_scaled = scaler.transform(X_flat)
        return X_scaled.reshape(shape)

    return transform(X_train), transform(X_val), transform(X_test), scaler


def compute_fold_metrics(y_true, preds, probs):
    """Compute accuracy / precision / recall / f1 / roc_auc for a fold."""

    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, average="binary", zero_division=0)
    recall = recall_score(y_true, preds, average="binary", zero_division=0)
    f1 = f1_score(y_true, preds, average="binary", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, probs[:, 1])
    except ValueError:
        # Happens if a fold's y_true has only one class present.
        roc_auc = np.nan

    return accuracy, precision, recall, f1, roc_auc


# --------------------------------------------------
# Main LOSO training loop
# --------------------------------------------------

def main():

    ensure_dirs()

    X, y, person_ids, feature_cols, encoders = preprocessing_pipeline(DATA_PATH)

    unique_people = np.unique(person_ids)
    num_classes = len(np.unique(y))
    input_shape = (X.shape[1], X.shape[2])

    all_fold_metrics = []

    # Track best validation ROC-AUC per architecture so we only keep
    # the single best checkpoint on disk for each model type.
    best_val_score = {name: -np.inf for name in MODEL_BUILDERS}
    best_val_path = {name: None for name in MODEL_BUILDERS}

    for model_name, build_fn in MODEL_BUILDERS.items():

        print("\n" + "=" * 60)
        print(f"Model: {model_name}")
        print("=" * 60)

        for held_out in unique_people:

            print(f"\n--- {model_name} | held-out participant: {held_out} ---")

            # LOSO split
            train_mask = person_ids != held_out
            test_mask = person_ids == held_out

            X_train_full = X[train_mask]
            y_train_full = y[train_mask]

            X_test = X[test_mask]
            y_test = y[test_mask]

            # Validation split (from training participants only)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=0.2,
                stratify=y_train_full,
                random_state=42,
            )

            # Scale inside the fold (no leakage) 
            X_train, X_val, X_test_scaled, scaler = scale_fold(X_train, X_val, X_test)

            # Class weights
            classes = np.unique(y_train)
            weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_train,
            )
            class_weight = dict(zip(classes, weights))

            # Build & compile
            model = build_fn(input_shape, num_classes)

            model.compile(
                optimizer=Adam(LEARNING_RATE),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            fold_checkpoint_path = os.path.join(
                SAVED_MODELS_DIR, f"{model_name}_{held_out}.keras"
            )

            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                ),
                ModelCheckpoint(
                    fold_checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                ),
            ]

            # Train
            model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1,
            )

            # Validation metrics (used to pick best checkpoint)
            val_probs = model.predict(X_val)
            val_preds = np.argmax(val_probs, axis=1)
            _, _, _, _, val_roc_auc = compute_fold_metrics(y_val, val_preds, val_probs)

            val_score = val_roc_auc if not np.isnan(val_roc_auc) else 0.0

            if val_score > best_val_score[model_name]:
                # Remove the previous best checkpoint for this architecture.
                prev_path = best_val_path[model_name]
                if prev_path is not None and prev_path != fold_checkpoint_path and os.path.exists(prev_path):
                    os.remove(prev_path)

                best_val_score[model_name] = val_score
                best_val_path[model_name] = fold_checkpoint_path
            else:
                # Not the best fold for this architecture -> discard checkpoint.
                if os.path.exists(fold_checkpoint_path):
                    os.remove(fold_checkpoint_path)

            # Test evaluation
            probs = model.predict(X_test_scaled)
            preds = np.argmax(probs, axis=1)

            accuracy, precision, recall, f1, roc_auc = compute_fold_metrics(y_test, preds, probs)
            cm = confusion_matrix(y_test, preds)

            print(f"Accuracy={accuracy:.4f}  Precision={precision:.4f}  "
                  f"Recall={recall:.4f}  F1={f1:.4f}  ROC-AUC={roc_auc:.4f}")
            print("Confusion matrix:")
            print(cm)

            all_fold_metrics.append({
                "Model": model_name,
                "HeldOut": held_out,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC_AUC": roc_auc,
            })

    # Rename best checkpoints to a stable name
    for model_name, path in best_val_path.items():
        if path is None:
            continue
        best_path = os.path.join(SAVED_MODELS_DIR, f"{model_name}_best.keras")
        if os.path.exists(path) and path != best_path:
            if os.path.exists(best_path):
                os.remove(best_path)
            os.rename(path, best_path)

    # Aggregate & save 
    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(METRICS_CSV, index=False)

    print("\n" + "=" * 60)
    print("LOSO Summary (mean ± std across folds)")
    print("=" * 60)

    summary = metrics_df.groupby("Model")[
        ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    ].agg(["mean", "std"])

    for model_name in MODEL_BUILDERS:
        if model_name not in summary.index:
            continue
        row = summary.loc[model_name]
        print(f"\n{model_name}")
        print(f"  Accuracy  : {row[('Accuracy', 'mean')]:.4f} ± {row[('Accuracy', 'std')]:.4f}")
        print(f"  Precision : {row[('Precision', 'mean')]:.4f} ± {row[('Precision', 'std')]:.4f}")
        print(f"  Recall    : {row[('Recall', 'mean')]:.4f} ± {row[('Recall', 'std')]:.4f}")
        print(f"  F1        : {row[('F1', 'mean')]:.4f} ± {row[('F1', 'std')]:.4f}")
        print(f"  ROC-AUC   : {row[('ROC_AUC', 'mean')]:.4f} ± {row[('ROC_AUC', 'std')]:.4f}")

    print(f"\nSaved per-fold metrics to {METRICS_CSV}")
    print(f"Saved best model per architecture to {SAVED_MODELS_DIR}/<name>_best.keras")


if __name__ == "__main__":
    main()