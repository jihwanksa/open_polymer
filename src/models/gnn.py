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
    
    def __init__(self, node_features=9, hidden_dim=128, num_layers=3, 
                 num_targets=5, dropout=0.2, gnn_type='gcn'):
        super(MoleculeGNN, self).__init__()
        
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        
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
        
        # Readout and prediction layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for mean + max pooling
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, num_targets)
    
    def forward(self, data):
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
    
    # Node features (atoms)
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
        ]
        
        features = atom_type + [
            atom.GetTotalDegree() / 4.0,
            atom.GetFormalCharge(),
            float(atom.GetIsAromatic()),
        ]
        atom_features.append(features)
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Edge index (bonds)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # Add reverse edge
    
    if len(edge_indices) == 0:
        # Isolated atom - create self-loop
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)


class GNNModel:
    """Wrapper for GNN training and inference"""
    
    def __init__(self, hidden_dim=128, num_layers=3, num_targets=5, 
                 gnn_type='gcn', device='cuda'):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.gnn_type = gnn_type
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
            node_features=9,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_targets=self.num_targets,
            gnn_type=self.gnn_type
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
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
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
                
                out = self.model(batch)
                
                # Compute loss only for non-zero targets
                mask = batch.y != 0
                if mask.sum() > 0:
                    loss = F.mse_loss(out[mask], batch.y[mask])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
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
                out = self.model(batch)
                
                mask = batch.y != 0
                if mask.sum() > 0:
                    loss = F.mse_loss(out[mask], batch.y[mask])
                    total_loss += loss.item()
                
                all_preds.append(out.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
        
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

