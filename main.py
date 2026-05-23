import yaml
import argparse
import pandas as pd
from src.data_loader import preprocess_data, get_loso_dataloaders
from src.model import DeepFlowCNN
from src.trainer import Trainer

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print("Preprocessing data...")
    df, subjects = preprocess_data(config)
    
    results = []
    
    print(f"Starting Leave-One-Subject-Out Cross-Validation on {len(subjects)} subjects...")
    for sub in subjects:
        print(f"\n--- Testing on Subject: {sub} ---")
        train_loader, test_loader = get_loso_dataloaders(df, sub, config)
        
        if len(test_loader.dataset) == 0:
            print(f"Skipping subject {sub} due to lack of data windows.")
            continue
            
        model = DeepFlowCNN(config)
        trainer = Trainer(model, config, sub)
        
        acc, f1 = trainer.fit(train_loader, test_loader)
        results.append({'subject': sub, 'accuracy': acc, 'f1': f1})
        
    res_df = pd.DataFrame(results)
    print("\n" + "="*30)
    print("FINAL RESULTS (LOSO)")
    print(res_df)
    print(f"Mean Accuracy: {res_df['accuracy'].mean():.4f}")
    print(f"Mean F1 Score: {res_df['f1'].mean():.4f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    main(args.config)
