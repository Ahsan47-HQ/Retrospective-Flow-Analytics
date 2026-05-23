import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

class Trainer:
    def __init__(self, model, config, subject_id):
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.subject_id = subject_id
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['lr'])
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize WandB run if it's the first fold or if we want separate runs
        # Usually one project with different fold tags is better
        self.run = wandb.init(
            project=config['project_name'],
            name=f"{config['experiment_name']}_subj_{subject_id}",
            config=config,
            reinit=True,
            tags=["LOSO", f"subject_{subject_id}"]
        )

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for x, meta, y in loader:
            x, meta, y = x.to(self.device), meta.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x, meta)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
        return total_loss / len(loader), accuracy_score(all_targets, all_preds)

    def evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, meta, y in loader:
                x, meta, y = x.to(self.device), meta.to(self.device), y.to(self.device)
                outputs = self.model(x, meta)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        return acc, f1

    def fit(self, train_loader, test_loader):
        epochs = self.config['training']['epochs']
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            test_acc, test_f1 = self.evaluate(test_loader)
            
            print(f"Subject {self.subject_id} | Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "subject": self.subject_id
            })
            
        self.run.finish()
        return test_acc, test_f1
