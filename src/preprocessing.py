import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(path):
    return pd.read_csv(path)



def clean_data(df):

    df = df.copy()

    df.dropna(inplace=True)

    return df



def encode_features(df):

    df = df.copy()

    encoders = {}

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns


    for col in categorical_cols:

        if col == "Person":
            continue


        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col]
        )

        encoders[col] = encoder


    return df, encoders



def scale_features(df, feature_cols):

    df = df.copy()

    scaler = StandardScaler()

    df[feature_cols] = scaler.fit_transform(
        df[feature_cols]
    )

    return df, scaler



def create_sequences(
    df,
    feature_cols,
    target,
    window_size=30
):

    X_seq = []
    y_seq = []


    for person, group in df.groupby("Person"):

        group = group.reset_index(drop=True)


        X = group[feature_cols].values

        y = group[target].values


        # non overlapping windows
        for i in range(
            0,
            len(group)-window_size,
            window_size
        ):

            X_seq.append(
                X[i:i+window_size]
            )


            # label of current window
            y_seq.append(
                y[i+window_size-1]
            )


    return (
        np.array(X_seq),
        np.array(y_seq)
    )




def split_data(X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(

        X,
        y,

        test_size=0.3,

        random_state=42,

        stratify=y
    )



    X_val, X_test, y_val, y_test = train_test_split(

        X_temp,
        y_temp,

        test_size=0.5,

        random_state=42,

        stratify=y_temp
    )


    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )




def preprocessing_pipeline(
    path,
    target="Correct",
    window_size=30
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

        "Puzzle",
        "DifficultyLevel"

    ]


    feature_cols = [

        col for col in df.columns

        if col not in remove_cols

    ]



    print("Features used:")
    print(feature_cols)



    df, scaler = scale_features(
        df,
        feature_cols
    )



    X, y = create_sequences(

        df,

        feature_cols,

        target,

        window_size

    )


    return (*split_data(X,y), scaler, encoders)





if __name__ == "__main__":


    (
        X_train,
        X_val,
        X_test,

        y_train,
        y_val,
        y_test,

        _,
        _

    ) = preprocessing_pipeline(

        "../data/raw/AllCombined_MentalEffortFocus.csv"

    )



    os.makedirs(
        "../data/processed",
        exist_ok=True
    )



    np.save("../data/processed/X_train.npy", X_train)
    np.save("../data/processed/X_val.npy", X_val)
    np.save("../data/processed/X_test.npy", X_test)


    np.save("../data/processed/y_train.npy", y_train)
    np.save("../data/processed/y_val.npy", y_val)
    np.save("../data/processed/y_test.npy", y_test)



    print("\nDONE")

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)