"""
Train BERT prediction heads for polymer properties.

This script:
1. Loads BERT model pretrained on SMILES
2. Extracts [CLS] embeddings for training data
3. Trains linear heads for each of the 5 properties
4. Saves the trained heads

These heads can then be used with generate_with_bert.py to create pseudo-labels.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForPreTraining
from pathlib import Path
from tqdm import tqdm

def get_bert_embeddings(smiles_list, tokenizer, model, device='cpu', batch_size=32):
    """Get [CLS] embeddings from BERT for SMILES."""
    print(f"Generating BERT embeddings for {len(smiles_list)} samples...")
    
    model = model.to(device)
    model.eval()
    
    embeddings = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        
        # Tokenize
        tokens = tokenizer(batch_smiles, padding=True, truncation=True, return_tensors='pt')
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model.bert(**tokens)
            # Get [CLS] token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        embeddings.append(cls_embeddings.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)
    print(f"✅ Generated embeddings shape: {embeddings.shape}")
    return embeddings


def train_linear_head(embeddings, targets, property_name, epochs=10, device='cpu'):
    """Train a linear head for a specific property."""
    print(f"\nTraining head for {property_name}...")
    
    # Filter out NaN targets
    valid_mask = ~np.isnan(targets)
    embeddings = embeddings[valid_mask]
    targets = targets[valid_mask]
    
    if len(targets) == 0:
        print(f"  ⚠️ No valid targets for {property_name}")
        return None
    
    print(f"  Using {len(targets)} valid samples")
    
    # Create dataset
    dataset = TensorDataset(embeddings, torch.FloatTensor(targets).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    head = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 1)
    ).to(device)
    
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Train
    head.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_embeddings, batch_targets in loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = head(batch_embeddings)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
    
    head.eval()
    print(f"  ✅ Head trained for {property_name}")
    
    return head


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv('data/raw/train.csv')
    print(f"Loaded {len(train_df)} training samples\n")
    
    # Load BERT
    print("Loading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained("unikei/bert-base-smiles")
    model = AutoModelForPreTraining.from_pretrained("unikei/bert-base-smiles")
    print("✅ BERT loaded\n")
    
    # Get embeddings
    embeddings = get_bert_embeddings(
        train_df['SMILES'].tolist(), 
        tokenizer, 
        model, 
        device=device,
        batch_size=32
    )
    
    # Train heads for each property
    heads = {}
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    for prop in properties:
        if prop in train_df.columns:
            head = train_linear_head(
                embeddings, 
                train_df[prop].values, 
                prop,
                epochs=10,
                device=device
            )
            if head is not None:
                heads[prop] = head
    
    # Save heads
    print("\nSaving trained heads...")
    save_dir = Path('models/bert_heads')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for prop, head in heads.items():
        torch.save(head.state_dict(), save_dir / f'{prop}_head.pt')
        print(f"  ✅ Saved {prop} head")
    
    print(f"\n{'='*60}")
    print("✅ BERT HEADS TRAINED!")
    print(f"{'='*60}")
    print(f"\nHeads saved in: {save_dir}")
    print(f"Ready to use with: python pseudolabel/generate_with_bert.py")


if __name__ == "__main__":
    main()

