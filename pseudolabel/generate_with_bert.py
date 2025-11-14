"""
Generate pseudo-labels using BERT + trained prediction heads.

Prerequisites:
1. Run: python pseudolabel/train_bert_heads.py  (trains heads once)
2. Then: python pseudolabel/generate_with_bert.py  (generates pseudo-labels)

Usage:
    conda activate pseudolabel_env
    cd /Users/jihwan/Downloads/open_polymer
    python pseudolabel/generate_with_bert.py --input_data data/PI1M_50000_v2.1.csv
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
from transformers import AutoTokenizer, AutoModelForPreTraining
from pathlib import Path
from tqdm import tqdm
import argparse

def load_bert_heads(device='cpu'):
    """Load trained BERT prediction heads."""
    print("Loading trained BERT heads...")
    
    heads = {}
    head_dir = Path('models/bert_heads')
    
    if not head_dir.exists():
        print(f"❌ Heads not found at {head_dir}")
        print("   Run: python pseudolabel/train_bert_heads.py first")
        return None
    
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    for prop in properties:
        head_file = head_dir / f'{prop}_head.pt'
        if not head_file.exists():
            print(f"⚠️  {prop} head not found")
            continue
        
        # Recreate head architecture
        head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        ).to(device)
        
        # Load weights
        head.load_state_dict(torch.load(head_file, map_location=device))
        head.eval()
        heads[prop] = head
        print(f"  ✅ Loaded {prop} head")
    
    if not heads:
        print("❌ No heads loaded!")
        return None
    
    return heads


def predict_batch(smiles_batch, tokenizer, bert_model, heads, device='cpu'):
    """Generate predictions for a batch of SMILES."""
    # Tokenize
    tokens = tokenizer(smiles_batch, padding=True, truncation=True, return_tensors='pt')
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = bert_model.bert(**tokens)
        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    
    # Generate predictions with each head
    predictions = []
    for smiles in range(len(smiles_batch)):
        pred_dict = {}
        for prop, head in heads.items():
            with torch.no_grad():
                pred = head(embeddings[smiles:smiles+1]).cpu().numpy()[0, 0]
            pred_dict[prop] = float(pred)
        predictions.append(pred_dict)
    
    return predictions


def main(args):
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}\n")
    
    # Check input file
    if not Path(args.input_data).exists():
        print(f"❌ Input file not found: {args.input_data}")
        return
    
    # Load models
    print("="*70)
    print("PSEUDO-LABEL GENERATION WITH BERT")
    print("="*70)
    
    print("\nLoading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained("unikei/bert-base-smiles")
    bert_model = AutoModelForPreTraining.from_pretrained("unikei/bert-base-smiles")
    bert_model = bert_model.to(device)
    bert_model.eval()
    print("✅ BERT loaded")
    
    heads = load_bert_heads(device)
    if heads is None:
        return
    
    # Load SMILES
    print(f"\nLoading SMILES from {args.input_data}...")
    df = pd.read_csv(args.input_data, usecols=['SMILES'])
    smiles_list = df['SMILES'].tolist()
    print(f"✅ Loaded {len(smiles_list)} SMILES\n")
    
    # Generate predictions
    print("🔮 Generating predictions...")
    all_predictions = []
    batch_size = 32
    
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Batches"):
        batch_smiles = smiles_list[i:i+batch_size]
        batch_preds = predict_batch(batch_smiles, tokenizer, bert_model, heads, device)
        all_predictions.extend(batch_preds)
    
    print(f"✅ Generated {len(all_predictions)} predictions\n")
    
    # Apply Tg transformation
    print("🔧 Applying Tg transformation: (9/5) × Tg + 45...")
    for pred in all_predictions:
        pred['Tg'] = (9/5) * pred['Tg'] + 45
    
    # Save
    print(f"\n💾 Saving to {args.output_path}...")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg'])
        writer.writeheader()
        for smiles, pred_dict in zip(smiles_list, all_predictions):
            row = {'SMILES': smiles}
            row.update(pred_dict)
            writer.writerow(row)
    
    print(f"✅ Saved {len(smiles_list)} pseudo-labeled samples\n")
    
    print("="*70)
    print("✅ PSEUDO-LABEL GENERATION COMPLETE!")
    print("="*70)
    print(f"\nOutput: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo-labels with BERT")
    parser.add_argument("--input_data", type=str, default="data/PI1M_50000_v2.1.csv")
    parser.add_argument("--output_path", type=str, default="pseudolabel/pi1m_pseudolabels_bert.csv")
    
    args = parser.parse_args()
    main(args)

