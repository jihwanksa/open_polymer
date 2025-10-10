"""
Data preprocessing utilities for polymer property prediction
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MolecularDataProcessor:
    """Process SMILES strings into various molecular representations"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def clean_smiles(self, smiles):
        """Clean SMILES string by removing polymer markers"""
        if pd.isna(smiles):
            return None
        # Remove polymer markers * 
        cleaned = smiles.replace('*', '[H]')
        return cleaned
    
    def smiles_to_mol(self, smiles):
        """Convert SMILES to RDKit molecule object"""
        cleaned = self.clean_smiles(smiles)
        if cleaned is None:
            return None
        try:
            mol = Chem.MolFromSmiles(cleaned)
            return mol
        except:
            return None
    
    def compute_molecular_descriptors(self, smiles):
        """Compute molecular descriptors from SMILES"""
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None
        
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'FractionCsp3': Descriptors.FractionCSP3(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            'MolMR': Descriptors.MolMR(mol),
            'BertzCT': Descriptors.BertzCT(mol),
            'Chi0': Descriptors.Chi0(mol),
            'Chi1': Descriptors.Chi1(mol),
        }
        return descriptors
    
    def compute_morgan_fingerprint(self, smiles, radius=2, n_bits=2048):
        """Compute Morgan (circular) fingerprints"""
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return np.array(fp)
        except:
            return None
    
    def load_and_process_data(self, train_path, test_path, supplement_paths=None):
        """Load and process all data"""
        print("Loading data...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Load supplementary data if provided
        if supplement_paths:
            supplement_dfs = []
            for path in supplement_paths:
                try:
                    df = pd.read_csv(path)
                    supplement_dfs.append(df)
                    print(f"Loaded supplement: {path}")
                except Exception as e:
                    print(f"Error loading {path}: {e}")
        
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        
        # Target columns
        target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        # Check missing values
        print("\nMissing values in targets:")
        for col in target_cols:
            missing_pct = (train_df[col] == 0).sum() / len(train_df) * 100
            print(f"{col}: {missing_pct:.2f}%")
        
        return train_df, test_df, target_cols
    
    def create_descriptor_features(self, df):
        """Create descriptor-based features for all samples"""
        print("\nComputing molecular descriptors...")
        descriptor_list = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            desc = self.compute_molecular_descriptors(row['SMILES'])
            if desc is not None:
                descriptor_list.append(desc)
                valid_indices.append(idx)
        
        descriptor_df = pd.DataFrame(descriptor_list, index=valid_indices)
        print(f"Created {len(descriptor_df.columns)} descriptors for {len(descriptor_df)} valid molecules")
        
        return descriptor_df
    
    def create_fingerprint_features(self, df, radius=2, n_bits=2048):
        """Create fingerprint-based features"""
        print(f"\nComputing Morgan fingerprints (radius={radius}, bits={n_bits})...")
        fingerprints = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            fp = self.compute_morgan_fingerprint(row['SMILES'], radius, n_bits)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)
        
        fp_df = pd.DataFrame(fingerprints, index=valid_indices)
        fp_df.columns = [f'fp_{i}' for i in range(n_bits)]
        print(f"Created fingerprints for {len(fp_df)} valid molecules")
        
        return fp_df


def explore_data(train_path, test_path):
    """Explore the dataset"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    print("\nTrain columns:", train_df.columns.tolist())
    print("\nFirst few SMILES:")
    print(train_df['SMILES'].head(3))
    
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    print("\nTarget variable statistics:")
    for col in target_cols:
        non_zero = train_df[train_df[col] != 0][col]
        if len(non_zero) > 0:
            print(f"\n{col}:")
            print(f"  Available samples: {len(non_zero)} ({len(non_zero)/len(train_df)*100:.1f}%)")
            print(f"  Mean: {non_zero.mean():.4f}")
            print(f"  Std: {non_zero.std():.4f}")
            print(f"  Min: {non_zero.min():.4f}")
            print(f"  Max: {non_zero.max():.4f}")
    
    print("\n" + "=" * 80)
    

if __name__ == "__main__":
    # Explore data
    explore_data('/home/jihwanoh/chem/train.csv', '/home/jihwanoh/chem/test.csv')
    
    # Test preprocessing
    processor = MolecularDataProcessor()
    train_df, test_df, target_cols = processor.load_and_process_data(
        '/home/jihwanoh/chem/train.csv',
        '/home/jihwanoh/chem/test.csv'
    )
    
    # Create descriptors for a small sample
    print("\n" + "=" * 80)
    print("Testing feature extraction on sample...")
    sample_df = train_df.head(100)
    desc_features = processor.create_descriptor_features(sample_df)
    print(f"Descriptor features shape: {desc_features.shape}")
    print(f"Sample descriptors:\n{desc_features.head()}")

