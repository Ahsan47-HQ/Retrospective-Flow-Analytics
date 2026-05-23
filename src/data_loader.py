import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

class RFADataset(Dataset):
    def __init__(self, windows, metadata, targets):
        self.windows = torch.FloatTensor(windows)
        self.metadata = torch.FloatTensor(metadata)
        self.targets = torch.LongTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.windows[idx], self.metadata[idx], self.targets[idx]

def preprocess_data(config):
    df = pd.read_csv(config['data']['csv_path'])
    
    # 1. Handle Missing Values
    df = df.ffill().bfill()

    # 2. Encoding Categorical Meta-data
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    
    # Normalize sensors that aren't already Z-scored (ACC and Movement)
    sensor_cols = config['data']['sensor_cols']
    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

    subjects = df[config['data']['subject_col']].unique()
    return df, subjects

def create_windows(df, config):
    window_size = config['data']['window_size']
    step_size = config['data']['step_size']
    sensor_cols = config['data']['sensor_cols']
    metadata_cols = config['data']['metadata_cols'] + ['Gender'] # Added encoded Gender
    target_col = config['data']['target_col']

    windows = []
    meta_vecs = []
    targets = []

    # Windowing per subject to avoid mixing temporal data across subjects
    for subject in df[config['data']['subject_col']].unique():
        subj_df = df[df[config['data']['subject_col']] == subject].reset_index(drop=True)
        
        if len(subj_df) < window_size:
            continue
            
        for i in range(0, len(subj_df) - window_size + 1, step_size):
            win = subj_df.iloc[i : i + window_size][sensor_cols].values
            # Metadata is usually constant per puzzle/subject in this dataset snippet, 
            # but we take the most frequent or middle value in window
            meta = subj_df.iloc[i + window_size // 2][metadata_cols].values
            target = subj_df.iloc[i + window_size - 1][target_col] # Predict focus at end of window
            
            windows.append(win)
            meta_vecs.append(meta)
            targets.append(target)
            
    return np.array(windows), np.array(meta_vecs), np.array(targets)

def get_loso_dataloaders(df, test_subject, config):
    train_df = df[df[config['data']['subject_col']] != test_subject]
    test_df = df[df[config['data']['subject_col']] == test_subject]

    X_train, M_train, y_train = create_windows(train_df, config)
    X_test, M_test, y_test = create_windows(test_df, config)

    train_ds = RFADataset(X_train, M_train, y_train)
    test_ds = RFADataset(X_test, M_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False)

    return train_loader, test_loader
