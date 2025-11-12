"""
Transformer-based model for SMILES using pre-trained ChemBERTa
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SMILESDataset(Dataset):
    """Dataset for SMILES strings"""
    
    def __init__(self, smiles_list, targets=None, tokenizer=None, max_length=256):
        self.smiles_list = smiles_list
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        
        # Tokenize SMILES
        encoding = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        
        if self.targets is not None:
            item['targets'] = torch.tensor(self.targets[idx], dtype=torch.float)
        
        return item


class TransformerMoleculeModel(nn.Module):
    """Transformer model with regression head"""
    
    def __init__(self, model_name='seyonec/ChemBERTa-zinc-base-v1', 
                 num_targets=5, dropout=0.2, hidden_dim=256):
        super(TransformerMoleculeModel, self).__init__()
        
        # Load pre-trained transformer
        try:
            print(f"Loading transformer model: {model_name}")
            self.transformer = AutoModel.from_pretrained(model_name)
            transformer_dim = self.transformer.config.hidden_size
        except Exception as e:
            print(f"Warning: Could not load {model_name}")
            print(f"Error: {e}")
            raise RuntimeError(f"Could not load transformer model: {model_name}")
        
        print(f"Using transformer: {model_name}")
        
        # Regression head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(transformer_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, num_targets)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim // 2)
    
    def forward(self, input_ids, attention_mask):
        # Get transformer embeddings
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Regression head
        x = self.dropout(pooled_output)
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc_out(x)
        
        return x


class TransformerModel:
    """Wrapper for transformer training and inference"""
    
    def __init__(self, model_name='seyonec/ChemBERTa-zinc-base-v1', 
                 num_targets=5, hidden_dim=256, dropout=0.2, device='cuda'):
        self.model_name = model_name
        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Auto-detect best device: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        self.model = None
        self.tokenizer = None
        
        print(f"Using device: {self.device}")
    
    def prepare_tokenizer(self):
        """Load tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except:
            print(f"Warning: Could not load tokenizer for {self.model_name}, using fallback")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        return self.tokenizer
    
    def prepare_data(self, smiles_list, targets=None, max_length=256):
        """Prepare dataset"""
        if self.tokenizer is None:
            self.prepare_tokenizer()
        
        dataset = SMILESDataset(
            smiles_list, 
            targets=targets, 
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        return dataset
    
    def create_model(self):
        """Create transformer model"""
        self.model = TransformerMoleculeModel(
            model_name=self.model_name,
            num_targets=self.num_targets,
            dropout=self.dropout,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        return self.model
    
    def train(self, train_dataset, val_dataset, epochs=50, batch_size=16, lr=2e-5):
        """Train transformer model"""
        if self.model is None:
            self.create_model()
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer with different learning rates for transformer and head
        optimizer = torch.optim.AdamW([
            {'params': self.model.transformer.parameters(), 'lr': lr},
            {'params': self.model.fc1.parameters(), 'lr': lr * 10},
            {'params': self.model.fc2.parameters(), 'lr': lr * 10},
            {'params': self.model.fc_out.parameters(), 'lr': lr * 10},
        ], weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        self.best_model_state = None  # Initialize to avoid AttributeError
        
        print(f"\nTraining Transformer Model...")
        print("=" * 80)
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                
                # Compute loss only for non-NaN targets (handle sparse labels)
                mask = ~torch.isnan(targets)
                if mask.sum() > 0:
                    loss = F.mse_loss(outputs[mask], targets[mask])
                    
                    # Check for NaN loss before backward
                    if torch.isnan(loss):
                        print(f"\n⚠️  NaN loss detected at epoch {epoch+1}. Skipping this batch.")
                        continue
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                    train_batches += 1
            
            train_loss = train_loss / train_batches if train_batches > 0 else 0
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader)
            
            # Check for NaN validation loss
            if math.isnan(val_loss):
                print(f"\n⚠️  NaN validation loss at epoch {epoch+1}. Training may be unstable.")
                print("Consider: 1) Lower learning rate, 2) Check input data, 3) Reduce batch size")
                if self.best_model_state is None:
                    print("No valid model state saved yet. Stopping training.")
                    return float('inf')
                break
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model if available
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nBest validation loss: {best_val_loss:.4f}")
        else:
            print("\n⚠️  No valid model state was saved during training.")
        
        return best_val_loss
    
    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                mask = ~torch.isnan(targets)
                if mask.sum() > 0:
                    loss = F.mse_loss(outputs[mask], targets[mask])
                    total_loss += loss.item()
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # Compute metrics per target
        metrics = {}
        for i in range(self.num_targets):
            # Check for non-NaN values (not just non-zero)
            mask = ~np.isnan(all_targets[:, i])
            if mask.sum() > 0:
                y_true = all_targets[mask, i]
                y_pred = all_preds[mask, i]
                
                # Only compute if we have valid data
                if len(y_true) > 0 and not np.isnan(y_pred).any():
                    metrics[f'target_{i}'] = {
                        'mse': mean_squared_error(y_true, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                        'mae': mean_absolute_error(y_true, y_pred),
                        'r2': r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0,
                    }
        
        return avg_loss, metrics
    
    def predict(self, test_dataset, batch_size=16):
        """Predict on test data"""
        self.model.eval()
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                all_preds.append(outputs.cpu().numpy())
        
        return np.vstack(all_preds)
    
    def save(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'model_name': self.model_name,
                'num_targets': self.num_targets,
                'hidden_dim': self.hidden_dim,
            }
        }, path)
        print(f"Transformer model saved to {path}")
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint['config']
        
        self.model_name = config['model_name']
        self.num_targets = config['num_targets']
        self.hidden_dim = config['hidden_dim']
        
        self.prepare_tokenizer()
        self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Transformer model loaded from {path}")

