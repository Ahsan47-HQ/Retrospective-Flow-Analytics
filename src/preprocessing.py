import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


# --------------------------------------------------
# Load
# --------------------------------------------------

def load_data(path):
    return pd.read_csv(path)


# --------------------------------------------------
# Clean
# --------------------------------------------------

def clean_data(df):

    df = df.copy()

    df.dropna(inplace=True)

    return df


# --------------------------------------------------
# Encode categorical variables
# --------------------------------------------------

def encode_features(df):

    df = df.copy()

    encoders = {}

    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:

        # Keep Person as string
        if col == "Person":
            continue

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(df[col])

        encoders[col] = encoder

    return df, encoders


# --------------------------------------------------
# Sliding windows
# --------------------------------------------------

def create_sequences(
    df,
    feature_cols,
    target,
    window_size=30,
    stride=5
):

    X = []
    y = []
    person_ids = []

    for person, group in df.groupby("Person",sort=False):

        group = group.sort_index().reset_index(drop=True)

        features = group[feature_cols].values

        labels = group[target].values

        for start in range(
            0,
            len(group) - window_size + 1,
            stride
        ):

            end = start + window_size

            X.append(
                features[start:end]
            )

            # label = last timestep
            y.append(
                labels[end - 1]
            )

            person_ids.append(person)

    return (
        np.array(X, dtype=np.float32),
        np.array(y),
        np.array(person_ids)
    )


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def preprocessing_pipeline(
    path,
    target="Correct",
    window_size=30,
    stride=5
):

    df = load_data(path)

    df = clean_data(df)

    df, encoders = encode_features(df)

    remove_cols = [

        "Person",

        target,

        # leakage
        "FocusLevel",
        "Focus01",
        
        # Nominal variable
        "Puzzle",

    ]

    feature_cols = [

        c for c in df.columns

        if c not in remove_cols

    ]

    print("\nFeatures Used:")

    for f in feature_cols:
        print(f)

    X, y, person_ids = create_sequences(

        df,

        feature_cols,

        target,

        window_size,

        stride

    )

    print("\nDataset Summary")

    print("-------------------------")

    print("Sequences :", len(X))

    print("Window    :", window_size)

    print("Stride    :", stride)

    print("Shape      :", X.shape)

    print("Labels     :", np.bincount(y))

    print("Participants :", np.unique(person_ids))

    return (

        X,

        y,

        person_ids,

        feature_cols,

        encoders

    )


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    X, y, persons, features, _ = preprocessing_pipeline(

        "../data/raw/AllCombined_MentalEffortFocus.csv",

        target="Correct",

        window_size=30,

        stride=5

    )

    print()

    print(X.shape)

    print(y.shape)

    print(persons.shape)