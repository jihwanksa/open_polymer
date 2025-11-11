"""
Graph Neural Network models for molecular property prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
import numpy as np
from rdkit import Chem
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm


class MoleculeGNN(nn.Module):
    """Graph Neural Network for molecular property prediction"""
    
    def __init__(self, node_features=16, hidden_dim=128, num_layers=3, 
                 num_targets=5, dropout=0.2, gnn_type='gcn', edge_features=6, aux_features=0):
        super(MoleculeGNN, self).__init__()
        
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.aux_features = aux_features  # Chemistry features dimension
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if gnn_type == 'gcn':
            self.convs.append(GCNConv(node_features, hidden_dim))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(node_features, hidden_dim, heads=4, concat=True))
            hidden_dim = hidden_dim * 4
        
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Middle layers
        for _ in range(num_layers - 1):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                # For intermediate layers, use single head to keep dimension consistent
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Readout and prediction layers with chemistry features
        self.dropout = nn.Dropout(dropout)
        # Combine graph readout (hidden_dim * 2) with chemistry features (aux_features)
        combined_dim = hidden_dim * 2 + aux_features
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, num_targets)
    
    def forward(self, data, aux_features=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Concatenate with chemistry features if provided
        if aux_features is not None:
            x = torch.cat([x, aux_features], dim=1)
        
        # Prediction layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        
        return x


def smiles_to_graph(smiles, clean_polymer_markers=True):
    """Convert SMILES string to PyG graph"""
    if clean_polymer_markers:
        smiles = smiles.replace('*', '[H]')
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
    except:
        return None
    
    # Enhanced node features (atoms) using RDKit chemistry information
    atom_features = []
    for atom in mol.GetAtoms():
        # Atom type (one-hot for common elements)
        atomic_num = atom.GetAtomicNum()
        atom_type = [
            atomic_num == 6,  # C
            atomic_num == 7,  # N
            atomic_num == 8,  # O
            atomic_num == 9,  # F
            atomic_num == 15, # P
            atomic_num == 16, # S
            atomic_num == 1,  # H
        ]
        
        # Chemical properties from RDKit
        from rdkit.Chem import AllChem, Descriptors
        features = atom_type + [
            atom.GetTotalDegree() / 4.0,           # Connectivity
            atom.GetFormalCharge() / 4.0,          # Charge (normalized)
            float(atom.GetIsAromatic()),           # Aromaticity
            atom.GetTotalNumHs() / 4.0,            # Hydrogen count
            atom.GetExplicitValence() / 8.0,       # Valence
            float(atom.IsInRing()),                # Is in ring
            float(atom.GetHybridization() == Chem.HybridizationType.SP),   # SP hybridization
            float(atom.GetHybridization() == Chem.HybridizationType.SP2),  # SP2
            float(atom.GetHybridization() == Chem.HybridizationType.SP3),  # SP3
        ]
        atom_features.append(features)
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Enhanced edge features (bonds)
    edge_index_list = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Bond type encoding
        bond_type = bond.GetBondType()
        bond_features = [
            float(bond_type == Chem.BondType.SINGLE),
            float(bond_type == Chem.BondType.DOUBLE),
            float(bond_type == Chem.BondType.TRIPLE),
            float(bond_type == Chem.BondType.AROMATIC),
            float(bond.GetIsAromatic()),
            float(bond.IsInRing()),
        ]
        
        # Add both directions
        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
        edge_features.append(bond_features)
        edge_features.append(bond_features)  # Same features for reverse edge
    
    if len(edge_index_list) == 0:
        # Isolated atom - create self-loop
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    
    # Create graph with enhanced features
    # Always create edge_attr for consistent batching (even if empty)
    if len(edge_features) > 0:
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        # For isolated atoms, create empty edge attributes
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return graph_data


class GNNModel:
    """Wrapper for GNN training and inference"""
    
    def __init__(self, hidden_dim=128, num_layers=3, num_targets=5, 
                 gnn_type='gcn', dropout=0.2, device='cuda'):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        
    def prepare_data(self, smiles_list, targets=None):
        """Prepare graph data from SMILES"""
        graph_data = []
        valid_indices = []
        
        for idx, smiles in enumerate(tqdm(smiles_list, desc="Converting SMILES to graphs")):
            graph = smiles_to_graph(smiles)
            if graph is not None:
                if targets is not None:
                    graph.y = torch.tensor(targets[idx], dtype=torch.float)
                graph_data.append(graph)
                valid_indices.append(idx)
        
        return graph_data, valid_indices
    
    def create_model(self):
        """Create GNN model"""
        self.model = MoleculeGNN(
            node_features=16,  # Updated: now includes RDKit enhanced features
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            dropout=self.dropout,
            gnn_type=self.gnn_type,
            edge_features=6
        ).to(self.device)
        return self.model
    
    def train(self, train_graphs, val_graphs, epochs=100, batch_size=32, lr=0.001):
        """Train GNN model"""
        if self.model is None:
            self.create_model()
        
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        print(f"\nTraining GNN ({self.gnn_type.upper()})...")
        print("=" * 80)
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = self.model(batch)  # Shape: [batch_size, num_targets]
                
                # Reshape batch.y from [batch_size * num_targets] to [batch_size, num_targets]
                batch_size = out.shape[0]
                num_targets = out.shape[1]
                target = batch.y.view(batch_size, num_targets)
                
                # Handle NaN values in targets
                mask = ~torch.isnan(target)
                
                if mask.any():
                    # Use masked loss - only compute loss for non-NaN values
                    masked_out = torch.where(mask, out, torch.zeros_like(out))
                    masked_target = torch.where(mask, target, torch.zeros_like(target))
                    
                    # Compute MSE only on valid values
                    squared_error = (masked_out - masked_target) ** 2
                    loss = (squared_error * mask.float()).sum() / mask.float().sum()
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                else:
                    # Skip batch if all values are NaN
                    pass
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 5 == 0 or epoch < 3:  # More frequent updates
                print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
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
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        print(f"\nBest validation loss: {best_val_loss:.4f}")
        
        return best_val_loss
    
    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                out = self.model(batch)  # Shape: [batch_size, num_targets]
                
                # Reshape batch.y from [batch_size * num_targets] to [batch_size, num_targets]
                batch_size = out.shape[0]
                num_targets = out.shape[1]
                target = batch.y.view(batch_size, num_targets)
                
                # Handle NaN values in targets
                mask = ~torch.isnan(target)
                
                if mask.any():
                    # Use masked loss
                    masked_out = torch.where(mask, out, torch.zeros_like(out))
                    masked_target = torch.where(mask, target, torch.zeros_like(target))
                    
                    squared_error = (masked_out - masked_target) ** 2
                    loss = (squared_error * mask.float()).sum() / mask.float().sum()
                    total_loss += loss.item()
                
                all_preds.append(out.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # Compute metrics per target
        metrics = {}
        for i in range(self.num_targets):
            mask = all_targets[:, i] != 0
            if mask.sum() > 0:
                y_true = all_targets[mask, i]
                y_pred = all_preds[mask, i]
                
                metrics[f'target_{i}'] = {
                    'mse': mean_squared_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred),
                }
        
        return avg_loss, metrics
    
    def predict(self, test_graphs, batch_size=32):
        """Predict on test data"""
        self.model.eval()
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                all_preds.append(out.cpu().numpy())
        
        return np.vstack(all_preds)
    
    def save(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_targets': self.num_targets,
                'gnn_type': self.gnn_type,
            }
        }, path)
        print(f"GNN model saved to {path}")
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint['config']
        
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.num_targets = config['num_targets']
        self.gnn_type = config['gnn_type']
        
        self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"GNN model loaded from {path}")

