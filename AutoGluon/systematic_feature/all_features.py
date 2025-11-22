
"""
Final Custom Feature Extraction Script
We want to use:
- RDKit 2D features
- RDKit graph features
- custom graph features
- custom graph features based on radical positions
- custom features based on backbone delineation
- custom features based on polymer application
- interactions between features
- ratios between features
- fingerprints (multiple strategies)
- degree of polymerization features (MD simulation with ~600 atoms)
- packing and thermal property indicators
"""

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, rdmolops, AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
import networkx as nx
import numpy as np

# Suppress RDKit warnings (especially UFFTYPER warnings for polymer wildcards)
RDLogger.DisableLog('rdApp.*')

# Global constant: Target number of atoms in MD simulation
TARGET_ATOMS_MD = 600


def calc_fused_ring_count(mol):
    G = nx.Graph()
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    cycles = nx.cycle_basis(G)
    fused_count = 0
    for i in range(len(cycles)):
        for j in range(i+1, len(cycles)):
            if set(cycles[i]) & set(cycles[j]):  # 공유 원자 있으면 fused
                fused_count += 1
    return fused_count


def calc_graph_features(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    features = {}
    if nx.is_connected(G):
        features["graph_diameter"] = nx.diameter(G)
        features["avg_shortest_path"] = nx.average_shortest_path_length(G)
    else:
        features["graph_diameter"] = None
        features["avg_shortest_path"] = None

    cycles = nx.cycle_basis(G)
    # Note: num_cycles removed as it's identical to RDKit's CalcNumRings
    features["cycle_basis_count"] = len(cycles)

    ring_atoms = set([a for cycle in cycles for a in cycle])
    features["atoms_in_rings_ratio"] = len(ring_atoms) / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0
    features["atoms_outside_rings"] = mol.GetNumAtoms() - len(ring_atoms)

    ring_sizes = [len(c) for c in cycles]
    features["max_ring_size"] = max(ring_sizes) if ring_sizes else 0
    # Note: min_ring_size removed as duplicate of max_ring_size for simple molecules

    for k in range(3, 11):
        features[f"has_ring_{k}"] = any(len(c) == k for c in cycles)

    degrees = [d for _, d in G.degree()]
    features["degree_distribution_mean"] = np.mean(degrees)
    features["degree_distribution_std"] = np.std(degrees)

    features["clustering_coefficient_mean"] = nx.average_clustering(G)
    features["betweenness_centrality_mean"] = np.mean(list(nx.betweenness_centrality(G).values()))
    features["eccentricity_mean"] = np.mean(list(nx.eccentricity(G).values())) if nx.is_connected(G) else None
    features["graph_density"] = nx.density(G)

    features["fused_ring_count"] = calc_fused_ring_count(mol)

    return features

# -----------------------------
# Radical/star descriptors
# -----------------------------

def _star_indices(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']

def graph_star_distance(mol):
    stars = _star_indices(mol)
    if len(stars) != 2:
        return None
    dmat = rdmolops.GetDistanceMatrix(mol)
    return int(dmat[stars[0], stars[1]])

def rings_between_stars(mol):
    stars = _star_indices(mol)
    if len(stars) != 2:
        return None
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    path = Chem.rdmolops.GetShortestPath(mol, stars[0], stars[1])
    return sum(1 for r in rings if any(a in r for a in path))

def ecfp_similarity_stars(mol, radius=2, nBits=1024):
    stars = _star_indices(mol)
    if len(stars) != 2:
        return None
    try:
        # Try newer API first
        gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
        fp0 = gen.GetFingerprint(mol, fromAtoms=[stars[0]])
        fp1 = gen.GetFingerprint(mol, fromAtoms=[stars[1]])
    except AttributeError:
        # Fall back to older API
        from rdkit.Chem import AllChem
        fp0 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, fromAtoms=[stars[0]])
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, fromAtoms=[stars[1]])
    
    on0 = set(fp0.GetOnBits()); on1 = set(fp1.GetOnBits())
    inter = len(on0 & on1); union = len(on0 | on1)
    return inter / union if union > 0 else 0.0

def peri_flag(mol):
    stars = _star_indices(mol)
    if len(stars) != 2:
        return 0
    d = graph_star_distance(mol)
    if d is None:
        return 0
    nbs0 = set([n.GetIdx() for n in mol.GetAtomWithIdx(stars[0]).GetNeighbors()])
    nbs1 = set([n.GetIdx() for n in mol.GetAtomWithIdx(stars[1]).GetNeighbors()])
    close = len(nbs0 & nbs1) > 0
    return int(d <= 4 and close)

def radical_distance(mol):
    try:
        radical_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
        if len(radical_idxs) < 2:
            return 0
        path = rdmolops.GetShortestPath(mol, radical_idxs[0], radical_idxs[1])
        return len(path) - 1
    except:
        return 0

def extract_tc_positional_features(smiles):
    """
    Extract radical/star position features (REMOVED duplicate: radical_distance)
    
    Removed:
    - radical_distance (duplicate of graph_star_distance)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "graph_star_distance": -1,
            "rings_between_stars": -1,
            "ecfp_similarity_stars": 0.0,
            "peri_flag": 0,
        }
    return {
        "graph_star_distance": graph_star_distance(mol) or -1,
        "rings_between_stars": rings_between_stars(mol) or -1,
        "ecfp_similarity_stars": ecfp_similarity_stars(mol) or 0.0,
        "peri_flag": peri_flag(mol) or 0,
    }


def _is_rotatable_bond(bond):
    """Check if a bond is rotatable (compatible with different RDKit versions)"""
    try:
        return bond.IsRotor()
    except AttributeError:
        # Fallback: single bond, not in ring, not terminal
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return False
        if bond.IsInRing():
            return False
        # Check if either atom is terminal (degree 1)
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if begin_atom.GetDegree() == 1 or end_atom.GetDegree() == 1:
            return False
        return True

def calc_backbone_sidechain(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: None for k in [
            "backbone_length", "backbone_aromatic_fraction", "backbone_rotatable_bonds",
            "backbone_hbond_donors", "backbone_hbond_acceptors", "backbone_polarity_index",
            "backbone_aromatic_ratio_total", "backbone_ring_count", "backbone_flexibility_score",
            "sidechain_count", "sidechain_avg_size", "sidechain_spacing_std", "sidechain_polarity_index",
            "sidechain_aromatic_fraction", "sidechain_bulkiness_score", "sidechain_hbond_donors",
            "sidechain_ring_count", "sidechain_polarity_score"]}

    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
    backbone_atoms = set()
    if len(stars) >= 2:
        G = nx.Graph()
        for b in mol.GetBonds():
            G.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        path = nx.shortest_path(G, stars[0], stars[-1])
        backbone_atoms = set(path)
    else:
        backbone_atoms = set(range(mol.GetNumAtoms()))

    backbone_length = len(backbone_atoms)
    aromatic_count = sum(1 for idx in backbone_atoms if mol.GetAtomWithIdx(idx).GetIsAromatic())
    backbone_aromatic_fraction = aromatic_count / backbone_length if backbone_length > 0 else 0
    backbone_rotatable_bonds = sum(1 for b in mol.GetBonds() if b.GetBeginAtomIdx() in backbone_atoms and b.GetEndAtomIdx() in backbone_atoms and _is_rotatable_bond(b))
    # Fixed: Use GetTotalNumHs() to count implicit + explicit hydrogens
    backbone_hbond_donors = sum(1 for idx in backbone_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() in [7,8] and mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0)
    backbone_hbond_acceptors = sum(1 for idx in backbone_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() in [7,8,9])
    polarity_atoms = {7,8,9,16,17,35,53}
    backbone_polarity_index = sum(1 for idx in backbone_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() in polarity_atoms)/backbone_length if backbone_length>0 else 0
    backbone_aromatic_ratio_total = aromatic_count/mol.GetNumAtoms() if mol.GetNumAtoms()>0 else 0
    backbone_ring_count = sum(1 for ring in mol.GetRingInfo().AtomRings() if any(a in backbone_atoms for a in ring))
    backbone_flexibility_score = backbone_rotatable_bonds/backbone_length if backbone_length>0 else 0

    sidechain_atoms = [idx for idx in range(mol.GetNumAtoms()) if idx not in backbone_atoms and mol.GetAtomWithIdx(idx).GetAtomicNum()>1]
    sidechain_count = len(sidechain_atoms)
    sidechain_sizes=[]; visited=set()
    for idx in sidechain_atoms:
        if idx in visited: continue
        stack=[idx]; size=0
        while stack:
            a=stack.pop()
            if a in visited: continue
            visited.add(a); size+=1
            for n in mol.GetAtomWithIdx(a).GetNeighbors():
                if n.GetIdx() in sidechain_atoms and n.GetIdx() not in visited:
                    stack.append(n.GetIdx())
        sidechain_sizes.append(size)
    sidechain_avg_size=np.mean(sidechain_sizes) if sidechain_sizes else 0
    sidechain_bulkiness_score=sum(sidechain_sizes)
    attachment_positions=[]; backbone_list=list(backbone_atoms)
    for idx in sidechain_atoms:
        for n in mol.GetAtomWithIdx(idx).GetNeighbors():
            if n.GetIdx() in backbone_atoms:
                attachment_positions.append(backbone_list.index(n.GetIdx()))
    sidechain_spacing_std=np.std(attachment_positions) if attachment_positions else 0
    sidechain_polarity_index=sum(1 for idx in sidechain_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() in polarity_atoms)/len(sidechain_atoms) if sidechain_atoms else 0
    sidechain_aromatic_fraction=sum(1 for idx in sidechain_atoms if mol.GetAtomWithIdx(idx).GetIsAromatic())/len(sidechain_atoms) if sidechain_atoms else 0
    # Fixed: Use GetTotalNumHs() to count implicit + explicit hydrogens
    sidechain_hbond_donors=sum(1 for idx in sidechain_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() in [7,8] and mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0)
    sidechain_ring_count=sum(1 for ring in mol.GetRingInfo().AtomRings() if any(a in sidechain_atoms for a in ring))
    sidechain_polarity_score=sidechain_polarity_index*sidechain_bulkiness_score

    # backbone_planarity_score = (aromatic_atoms_in_backbone / backbone_length) - (rotatable_bonds_in_backbone / backbone_length)
    aromatic_atoms_in_backbone = sum(1 for idx in backbone_atoms if mol.GetAtomWithIdx(idx).GetIsAromatic())
    rotatable_bonds_in_backbone = sum(1 for b in mol.GetBonds() if b.GetBeginAtomIdx() in backbone_atoms and b.GetEndAtomIdx() in backbone_atoms and _is_rotatable_bond(b))
    backbone_planarity_score = (aromatic_atoms_in_backbone / backbone_length if backbone_length > 0 else 0) - (rotatable_bonds_in_backbone / backbone_length if backbone_length > 0 else 0)


    return {
        "backbone_length": backbone_length,
        "backbone_aromatic_fraction": backbone_aromatic_fraction,
        "backbone_rotatable_bonds": backbone_rotatable_bonds,
        "backbone_hbond_donors": backbone_hbond_donors,
        "backbone_hbond_acceptors": backbone_hbond_acceptors,
        "backbone_polarity_index": backbone_polarity_index,
        "backbone_aromatic_ratio_total": backbone_aromatic_ratio_total,
        "backbone_ring_count": backbone_ring_count,
        "backbone_flexibility_score": backbone_flexibility_score,
        "sidechain_count": sidechain_count,
        "sidechain_avg_size": sidechain_avg_size,
        "sidechain_spacing_std": sidechain_spacing_std,
        "sidechain_polarity_index": sidechain_polarity_index,
        "sidechain_aromatic_fraction": sidechain_aromatic_fraction,
        "sidechain_bulkiness_score": sidechain_bulkiness_score,
        "sidechain_hbond_donors": sidechain_hbond_donors,
        "sidechain_ring_count": sidechain_ring_count,
        "sidechain_polarity_score": sidechain_polarity_score,
        "backbone_planarity_score": backbone_planarity_score,
        "rigidity_index": backbone_planarity_score - backbone_flexibility_score
    }

# More custom features
def calc_paper_based_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: None for k in ["fluorine_fraction","sp2_sp3_ratio_all_atoms","functional_group_diversity"]}

    # Fluorine fraction (more specific than fr_halogen which counts all halogens)
    fluorine_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    fluorine_fraction = fluorine_count / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0

    # sp2/sp3 ratio (all atoms, not just carbons)
    # Note: Different from FractionCSP3 which is carbon-only fraction
    # This captures overall hybridization including heteroatoms
    sp2_count = sum(1 for a in mol.GetAtoms() if str(a.GetHybridization()) == "SP2")
    sp3_count = sum(1 for a in mol.GetAtoms() if str(a.GetHybridization()) == "SP3")
    sp2_sp3_ratio_all_atoms = sp2_count / (sp3_count if sp3_count > 0 else 1)

    # Count unique functional groups based on SMARTS patterns
    patterns = {
        "OH": "[OX2H]",       # Hydroxyl
        "NH2": "[NX3;H2]",    # Primary amine
        "COOH": "C(=O)[OH]",  # Carboxylic acid
        "NO2": "[N+](=O)[O-]",# Nitro
        "C=O": "C=O",         # Carbonyl
        "CN": "C#N"           # Nitrile
    }
    diversity_count = sum(1 for name, smarts in patterns.items() if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)))

    return {
        "fluorine_fraction": fluorine_fraction,
        "sp2_sp3_ratio_all_atoms": sp2_sp3_ratio_all_atoms,
        "functional_group_diversity": diversity_count
    }

# Interaction features

def calc_interactions(base_features):
    interactions={}
    if "TPSA" in base_features and "MolWt" in base_features:
        interactions["TPSA_MolWt_pair"]=(base_features["TPSA"],base_features["MolWt"])
    if "NumHDonors" in base_features and "NumHAcceptors" in base_features:
        interactions["NumHDonors_NumHAcceptors_pair"]=(base_features["NumHDonors"],base_features["NumHAcceptors"])
    if "FractionCSP3" in base_features and "MolLogP" in base_features:
        interactions["FractionCSP3_MolLogP_pair"]=(base_features["FractionCSP3"],base_features["MolLogP"])
    if "RingCount" in base_features and "MolMR" in base_features:
        interactions["RingCount_MolMR_pair"]=(base_features["RingCount"],base_features["MolMR"])
    if "backbone_aromatic_fraction" in base_features and "sidechain_hbond_donors" in base_features:
        interactions["backbone_aromatic_fraction_sidechain_hbond_donors_pair"]=(base_features["backbone_aromatic_fraction"],base_features["sidechain_hbond_donors"])
    if "backbone_length" in base_features and "sidechain_spacing_std" in base_features:
        interactions["backbone_length_sidechain_spacing_std_pair"]=(base_features["backbone_length"],base_features["sidechain_spacing_std"])
    return interactions

# Derived indices

def calc_derived(base_features):
    derived={}
    if "NumRotatableBonds" in base_features and "HeavyAtomCount" in base_features:
        derived["flexibility_index"]=base_features["NumRotatableBonds"]/(base_features["HeavyAtomCount"] or 1)
    if "TPSA" in base_features and "MolWt" in base_features:
        derived["polarity_ratio"]=base_features["TPSA"]/(base_features["MolWt"] or 1)
    if "NumAtoms" in base_features and "backbone_length" in base_features:
        derived["branching_index"]=base_features["NumAtoms"]/(base_features["backbone_length"] or 1)
    return derived

# Fingerprint PCA

def calc_fingerprints_all_strategies(mol, include_full_arrays=True):
    """
    Calculate fingerprints using ALL 4 strategies + full arrays for comprehensive coverage
    
    Args:
        mol: RDKit molecule
        include_full_arrays: If True, include full fingerprint arrays (adds ~2200 features)
    
    Returns:
    - Without arrays: ~108 features (counts, folded, hashed, statistical)
    - With arrays: ~2300 features (108 + 1024 Morgan + 1024 AtomPair + 1024 Torsion + 167 MACCS)
    """
    features = {}
    
    # Get fingerprints
    morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    morgan_arr = np.array(morgan_fp, dtype=int)
    
    # AtomPair: Use rdFingerprintGenerator (cleaner API)
    try:
        from rdkit.Chem import rdFingerprintGenerator
        atompair_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)
        atompair_fp = atompair_gen.GetFingerprint(mol)
        atompair_arr = np.array(atompair_fp, dtype=int)
    except:
        atompair_arr = np.zeros(1024, dtype=int)
    
    # Torsion: Use rdFingerprintGenerator (cleaner API)
    try:
        from rdkit.Chem import rdFingerprintGenerator
        torsion_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024)
        torsion_fp = torsion_gen.GetFingerprint(mol)
        torsion_arr = np.array(torsion_fp, dtype=int)
    except:
        torsion_arr = np.zeros(1024, dtype=int)
    
    try:
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.array(maccs_fp, dtype=int)
    except:
        maccs_arr = np.zeros(167, dtype=int)
    
    # Strategy 1: Counts (4 features)
    features['fp_Morgan_count'] = int(morgan_arr.sum())
    features['fp_AtomPair_count'] = int(atompair_arr.sum())
    features['fp_Torsion_count'] = int(torsion_arr.sum())
    features['fp_MACCS_count'] = int(maccs_arr.sum())
    
    # Strategy 2: Folded (64 features)
    n_bits = 64
    folded = morgan_arr.reshape(n_bits, -1).sum(axis=1) % 2
    for i in range(n_bits):
        features[f'fp_Morgan_fold{i}'] = int(folded[i])
    
    # Strategy 3: Hashed (32 features)
    morgan_hashed = AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=32)
    for i in range(32):
        features[f'fp_Morgan_hash{i}'] = morgan_hashed[i]
    
    # Strategy 4: Statistical (10 features)
    features['fp_Morgan_density'] = float(morgan_arr.mean())
    on_bits = np.where(morgan_arr == 1)[0]
    if len(on_bits) > 0:
        features['fp_Morgan_first_bit'] = int(on_bits[0])
        features['fp_Morgan_last_bit'] = int(on_bits[-1])
        features['fp_Morgan_mean_position'] = float(on_bits.mean())
        features['fp_Morgan_std_position'] = float(on_bits.std())
    else:
        features['fp_Morgan_first_bit'] = 0
        features['fp_Morgan_last_bit'] = 0
        features['fp_Morgan_mean_position'] = 0.0
        features['fp_Morgan_std_position'] = 0.0
    
    # MACCS statistics
    features['fp_MACCS_density'] = float(maccs_arr.mean())
    on_bits = np.where(maccs_arr == 1)[0]
    if len(on_bits) > 0:
        features['fp_MACCS_mean_position'] = float(on_bits.mean())
        features['fp_MACCS_std_position'] = float(on_bits.std())
    else:
        features['fp_MACCS_mean_position'] = 0.0
        features['fp_MACCS_std_position'] = 0.0
    
    # Strategy 5: Full arrays (2239 features) - Raw fingerprint bits
    if include_full_arrays:
        # Morgan (1024 bits)
        for i in range(len(morgan_arr)):
            features[f'fp_Morgan_bit{i}'] = int(morgan_arr[i])
        
        # AtomPair (1024 bits)
        for i in range(len(atompair_arr)):
            features[f'fp_AtomPair_bit{i}'] = int(atompair_arr[i])
        
        # Torsion (1024 bits)
        for i in range(len(torsion_arr)):
            features[f'fp_Torsion_bit{i}'] = int(torsion_arr[i])
        
        # MACCS (167 bits, bit 0 is always 0)
        for i in range(len(maccs_arr)):
            features[f'fp_MACCS_bit{i}'] = int(maccs_arr[i])
    
    return features

def calc_fingerprints(mol, use_full_fps=False):
    """
    Calculate multiple fingerprint types as full bit vectors
    
    Fingerprint types:
    - Morgan (ECFP): Circular fingerprints (1024 bits)
    - Atom Pair: Topological distance-based (1024 bits)
    - Topological Torsion: 4-atom path-based (1024 bits)
    - MACCS Keys: 166 structural keys
    
    Args:
        mol: RDKit molecule
        use_full_fps: If True, return all bits (1024 per FP). If False, return count-based features.
    
    Note: Full fingerprints (use_full_fps=True) will add ~3000 features!
    For most ML tasks, count-based features are sufficient.
    """
    from rdkit.Chem import AllChem, MACCSkeys
    from rdkit.Chem.AtomPairs import Pairs, Torsions
    
    features = {}
    
    if use_full_fps:
        # Return full bit vectors (WARNING: 3000+ features!)
        # 1. Morgan fingerprint (1024 bits)
        morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        morgan_arr = np.array(morgan_fp)
        for i in range(len(morgan_arr)):
            features[f"Morgan_r2_bit{i}"] = int(morgan_arr[i])
        
        # 2. Atom Pair fingerprint (1024 bits)
        try:
            atompair_fp = Pairs.GetAtomPairFingerprintAsBitVect(mol, nBits=1024)
            atompair_arr = np.array(atompair_fp)
            for i in range(len(atompair_arr)):
                features[f"AtomPair_bit{i}"] = int(atompair_arr[i])
        except:
            pass
        
        # 3. Topological Torsion fingerprint (1024 bits)
        try:
            torsion_fp = Torsions.GetTopologicalTorsionFingerprintAsBitVect(mol, nBits=1024)
            torsion_arr = np.array(torsion_fp)
            for i in range(len(torsion_arr)):
                features[f"Torsion_bit{i}"] = int(torsion_arr[i])
        except:
            pass
        
        # 4. MACCS Keys (167 bits, bit 0 is always 0)
        try:
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            maccs_arr = np.array(maccs_fp)
            for i in range(1, len(maccs_arr)):  # Skip bit 0
                features[f"MACCS_key{i}"] = int(maccs_arr[i])
        except:
            pass
    else:
        # Return count-based features (more compact, often sufficient)
        # Count of set bits in each fingerprint
        morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        features['Morgan_r2_count'] = morgan_fp.GetNumOnBits()
        
        try:
            atompair_fp = Pairs.GetAtomPairFingerprintAsBitVect(mol, nBits=1024)
            features['AtomPair_count'] = atompair_fp.GetNumOnBits()
        except:
            features['AtomPair_count'] = 0
        
        try:
            torsion_fp = Torsions.GetTopologicalTorsionFingerprintAsBitVect(mol, nBits=1024)
            features['Torsion_count'] = torsion_fp.GetNumOnBits()
        except:
            features['Torsion_count'] = 0
        
        try:
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            features['MACCS_count'] = maccs_fp.GetNumOnBits()
        except:
            features['MACCS_count'] = 0
    
    return features

# RDKit comprehensive extraction

def extract_all_rdkit_2d(mol):
    """Extract all ~208 RDKit 2D descriptors (with NaN/inf cleaning)"""
    from rdkit.Chem import Descriptors
    features = {}
    for name, func in Descriptors.descList:
        try:
            value = func(mol)
            # Clean NaN/inf values (Gasteiger charge calculation has numerical bugs)
            if isinstance(value, float):
                if np.isnan(value):
                    # NaN → None (let AutoGluon handle missing values)
                    value = None
                elif np.isinf(value):
                    # Inf → ±2.0 (reasonable bounds for partial charges, which max at ~0.7)
                    if 'PartialCharge' not in name:
                        print(f"Warning: Inf in non-charge descriptor: {name} (value: {value})")
                        print(f"   Clipping to ±2.0 may not be appropriate for this descriptor!")
                    value = 2.0 if value > 0 else -2.0
            features[f"rdkit_2d_{name}"] = value
        except:
            features[f"rdkit_2d_{name}"] = None
    return features

def extract_all_rdkit_graph(mol):
    """
    Extract RDKit graph descriptors (SKIPPING duplicates with 2D/Molecular)
    
    Skipped (duplicates of rdkit_2d_*):
    - BalabanJ, BertzCT, Chi0-4 (n/v), HallKierAlpha, Ipc, Kappa1-3
    
    Kept (unique):
    - ChiNn_, ChiNv_ (these return None but are in the module)
    """
    from rdkit.Chem import GraphDescriptors
    features = {}
    
    # Skip duplicates - these are already in rdkit_2d_*
    skip_duplicates = [
        'BalabanJ', 'BertzCT', 
        'Chi0', 'Chi0n', 'Chi0v',
        'Chi1', 'Chi1n', 'Chi1v',
        'Chi2n', 'Chi2v',
        'Chi3n', 'Chi3v',
        'Chi4n', 'Chi4v',
        'HallKierAlpha', 'Ipc',
        'Kappa1', 'Kappa2', 'Kappa3',
    ]
    
    graph_funcs = [attr for attr in dir(GraphDescriptors) 
                   if not attr.startswith('_') and callable(getattr(GraphDescriptors, attr))]
    
    for name in graph_funcs:
        if name in skip_duplicates:
            continue
        try:
            features[f"rdkit_graph_{name}"] = getattr(GraphDescriptors, name)(mol)
        except:
            features[f"rdkit_graph_{name}"] = None
    return features
"""
def extract_all_rdkit_3d(mol):
    ````
    Extract RDKit 3D descriptors (SKIPPING - all are duplicates of rdMolDescriptors.Calc*)
    
    Descriptors3D is just a wrapper for rdMolDescriptors.Calc* functions.
    All 10 descriptors are exact duplicates:
    - Asphericity == CalcAsphericity
    - Eccentricity == CalcEccentricity
    - InertialShapeFactor == CalcInertialShapeFactor
    - NPR1 == CalcNPR1, NPR2 == CalcNPR2
    - PMI1 == CalcPMI1, PMI2 == CalcPMI2, PMI3 == CalcPMI3
    - RadiusOfGyration == CalcRadiusOfGyration
    - SpherocityIndex == CalcSpherocityIndex
    
    We keep the Calc* versions in extract_rdkit_molecular_descriptors()
    ```
    # Return empty dict - all 3D descriptors are in rdMolDescriptors.Calc*
    return {}
"""
def generate_3d_conformer(mol):
    """
    Generate 3D conformer using ETKDG and optimize with UFF
    
    For polymer repeat units with radical atoms (*), we:
    1. Replace * with H temporarily
    2. Generate 3D conformer
    3. Optimize with UFF
    4. Copy conformer back to original molecule
    
    Args:
        mol: RDKit molecule (will be modified in place)
    
    Returns:
        bool: True if successful, False otherwise
    """
    from rdkit.Chem import AllChem
    
    try:
        # Check for radical atoms (*)
        has_radicals = any(a.GetSymbol() == '*' for a in mol.GetAtoms())
        
        if has_radicals:
            # Replace * with C for 3D generation (more realistic for polymer chains)
            mol_copy = Chem.RWMol(mol)
            radical_indices = []
            for atom in mol_copy.GetAtoms():
                if atom.GetSymbol() == '*':
                    radical_indices.append(atom.GetIdx())
                    atom.SetAtomicNum(6)  # Replace with C
            
            mol_for_3d = mol_copy.GetMol()
        else:
            mol_for_3d = mol
        
        # Add hydrogens
        mol_h = Chem.AddHs(mol_for_3d)
        
        # Generate 3D conformer using ETKDG
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  # For reproducibility
        result = AllChem.EmbedMolecule(mol_h, params)
        
        if result == -1:
            return False
        
        # Optimize with UFF force field
        try:
            AllChem.UFFOptimizeMolecule(mol_h)
        except:
            # UFF might fail, but we can still use the unoptimized conformer
            pass
        
        # Copy conformer back to original molecule
        if mol_h.GetNumConformers() > 0:
            conf = mol_h.GetConformer()
            # Create conformer for original mol
            new_conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                new_conf.SetAtomPosition(i, pos)
            mol.AddConformer(new_conf)
            return True
        
        return False
    except Exception as e:
        return False

def extract_rdkit_molecular_descriptors(mol, include_3d=True):
    """
    Extract unique Calc* functions from rdMolDescriptors
    
    Skipped duplicates with rdkit_2d_*:
    - CalcChi0n, CalcChi0v, CalcChi1n, CalcChi1v, CalcChi2n, CalcChi2v
    - CalcChi3n, CalcChi3v, CalcChi4n, CalcChi4v
    - CalcHallKierAlpha, CalcKappa1, CalcKappa2, CalcKappa3
    - CalcLabuteASA, CalcTPSA, CalcFractionCSP3
    - CalcExactMolWt, CalcNumHeavyAtoms, CalcNumHeteroatoms
    - CalcNumRotatableBonds
    - CalcNumAromaticCarbocycles, CalcNumAromaticRings, CalcNumRings
    
    3D descriptors (11) - included if include_3d=True:
    - CalcAsphericity, CalcEccentricity, CalcInertialShapeFactor
    - CalcNPR1, CalcNPR2, CalcPMI1, CalcPMI2, CalcPMI3
    - CalcPBF, CalcRadiusOfGyration, CalcSpherocityIndex
    
    Kept unique (19 without 3D, 30 with 3D):
    - CalcNumHBA, CalcNumHBD
    - CalcNumLipinskiHBA, CalcNumLipinskiHBD
    - CalcNumAmideBonds, CalcNumBridgeheadAtoms, CalcNumSpiroAtoms
    - CalcNumAtomStereoCenters, CalcNumUnspecifiedAtomStereoCenters
    - CalcNumHeterocycles, CalcPhi, CalcNumAtoms
    - CalcNumAliphaticCarbocycles, CalcNumAliphaticHeterocycles, CalcNumAliphaticRings
    - CalcNumSaturatedCarbocycles, CalcNumSaturatedHeterocycles, CalcNumSaturatedRings
    """
    features = {}
    
    # Get all Calc* functions
    calc_funcs = [name for name in dir(rdMolDescriptors) 
                  if name.startswith('Calc') and callable(getattr(rdMolDescriptors, name))]
    
    # Functions that return arrays/matrices/strings
    skip_funcs = [
        'CalcAUTOCORR2D', 'CalcAUTOCORR3D',  # Return arrays
        'CalcCoulombMat',  # Returns matrix
        'CalcGETAWAY', 'CalcMORSE', 'CalcRDF', 'CalcWHIM',  # Return arrays
        'CalcCrippenDescriptors',  # Returns tuple
        'CalcEEMcharges',  # Returns array
        'CalcMolFormula',  # Returns string
        'CalcChiNn', 'CalcChiNv',  # Need additional parameters
    ]
    
    # Duplicates with rdkit_2d_*
    skip_duplicates_2d = [
        'CalcChi0n', 'CalcChi0v', 'CalcChi1n', 'CalcChi1v',
        'CalcChi2n', 'CalcChi2v', 'CalcChi3n', 'CalcChi3v',
        'CalcChi4n', 'CalcChi4v',
        'CalcHallKierAlpha', 'CalcKappa1', 'CalcKappa2', 'CalcKappa3',
        'CalcLabuteASA', 'CalcTPSA', 'CalcFractionCSP3',
        'CalcExactMolWt', 'CalcNumHeavyAtoms', 'CalcNumHeteroatoms',
        'CalcNumRotatableBonds',
        'CalcNumAromaticCarbocycles', 'CalcNumAromaticRings', 'CalcNumRings',
    ]
    
    # 3D descriptors (include if requested and conformer generation succeeds)
    descriptors_3d = [
        'CalcAsphericity', 'CalcEccentricity', 'CalcInertialShapeFactor',
        'CalcNPR1', 'CalcNPR2', 'CalcPMI1', 'CalcPMI2', 'CalcPMI3',
        'CalcPBF', 'CalcRadiusOfGyration', 'CalcSpherocityIndex',
    ]
    
    skip_funcs.extend(skip_duplicates_2d)
    
    # Generate 3D conformer if needed
    if include_3d:
        success = generate_3d_conformer(mol)
        if not success:
            # If 3D generation fails, skip 3D descriptors
            skip_funcs.extend(descriptors_3d)
    else:
        skip_funcs.extend(descriptors_3d)
    
    for func_name in calc_funcs:
        if func_name in skip_funcs:
            continue
        try:
            func = getattr(rdMolDescriptors, func_name)
            result = func(mol)
            # Only keep scalar values
            if isinstance(result, (int, float)):
                features[f"rdkit_mol_{func_name}"] = result
        except:
            features[f"rdkit_mol_{func_name}"] = None
    
    return features

# Lipinski module removed - ALL functions are duplicates:
# - NumHDonors/NumHAcceptors are in rdkit_2d_* (NHOHCount, NOCount)
# - CalcNumLipinskiHBA/HBD are in rdMolDescriptors
# - All other functions duplicate rdkit_2d_* descriptors

# Main generator

def generate_custom_features(smiles, base_features):
    """Generate only custom polymer-specific features (no RDKit duplicates)"""
    mol = Chem.MolFromSmiles(smiles)
    features = {}
    features.update(calc_backbone_sidechain(smiles))
    features.update(calc_interactions({**base_features, **features}))
    features.update(calc_derived({**base_features, **features}))
    features.update(calc_fingerprint_pca(mol))
    return features

def calc_degree_of_polymerization(smiles, target_atoms=TARGET_ATOMS_MD):
    """
    Calculate degree of polymerization assuming target number of atoms from MD simulation
    
    Critical for properties that scale with chain length (Tg, Rg, density, FFV, thermal conductivity)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            'dp_degree_of_polymerization': 0,
            'dp_estimated_molecular_weight': 0,
            'dp_atoms_per_repeat_unit': 0,
            'dp_mw_per_repeat_unit': 0,
            'dp_estimated_backbone_length_angstrom': 0,
        }
    
    # Count atoms in repeat unit (excluding radicals *)
    atoms_per_unit = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    
    if atoms_per_unit == 0:
        return {
            'dp_degree_of_polymerization': 0,
            'dp_estimated_molecular_weight': 0,
            'dp_atoms_per_repeat_unit': 0,
            'dp_mw_per_repeat_unit': 0,
            'dp_estimated_backbone_length_angstrom': 0,
        }
    
    # Calculate degree of polymerization
    dp = target_atoms / atoms_per_unit
    
    # Molecular weight of repeat unit
    mw_per_unit = sum(a.GetMass() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    
    # Estimated total molecular weight
    estimated_mw = mw_per_unit * dp
    
    # Chain length estimate (assuming extended conformation, C-C bond ~1.54 Å)
    backbone_carbons = sum(1 for a in mol.GetAtoms() 
                          if a.GetAtomicNum() == 6 and a.GetSymbol() != '*')
    estimated_backbone_length = backbone_carbons * dp * 1.54  # Angstroms
    
    return {
        'dp_degree_of_polymerization': dp,
        'dp_estimated_molecular_weight': estimated_mw,
        'dp_atoms_per_repeat_unit': atoms_per_unit,
        'dp_mw_per_repeat_unit': mw_per_unit,
        'dp_estimated_backbone_length_angstrom': estimated_backbone_length,
    }

def calc_packing_and_thermal_features(smiles):
    """
    Features for density, FFV, Tg, and thermal conductivity prediction
    
    Returns 9 unique features (no duplicates with existing features)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    features = {}
    
    # Get DP for calculations
    dp_features = calc_degree_of_polymerization(smiles)
    atoms_per_unit = dp_features['dp_atoms_per_repeat_unit']
    mw_per_unit = dp_features['dp_mw_per_repeat_unit']
    
    # 1. Compactness: MW per atom (heavier atoms → denser packing)
    if atoms_per_unit > 0:
        features['packing_compactness_mw_per_atom'] = mw_per_unit / atoms_per_unit
    else:
        features['packing_compactness_mw_per_atom'] = 0
    
    # 2. Chain flexibility per atom (affects Tg, FFV)
    rotatable_bonds = sum(1 for b in mol.GetBonds() 
                         if b.GetBondType() == Chem.BondType.SINGLE 
                         and not b.IsInRing())
    if atoms_per_unit > 0:
        features['thermal_flexibility_per_atom'] = rotatable_bonds / atoms_per_unit
    else:
        features['thermal_flexibility_per_atom'] = 0
    
    # 3. Aromatic content (whole molecule, not just backbone)
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    total_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    features['thermal_aromatic_fraction'] = aromatic_atoms / total_atoms if total_atoms > 0 else 0
    
    # 4. Polar groups (affect Tg through H-bonding)
    polar_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 9, 16])
    features['thermal_polar_atom_fraction'] = polar_atoms / total_atoms if total_atoms > 0 else 0
    
    # 5. Atom type diversity (symmetry indicator, affects packing/crystallinity)
    # Note: Sidechain/backbone ratio removed as duplicate (can be derived from sidechain_count / backbone_length)
    atom_types = [a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(atom_types) > 0:
        features['packing_atom_type_diversity'] = len(set(atom_types)) / len(atom_types)
    else:
        features['packing_atom_type_diversity'] = 0
    
    # 7-9. Van der Waals volume features (for FFV, density)
    vdw_radii = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 
                 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85}
    
    total_vdw_volume = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            radius = vdw_radii.get(atom.GetAtomicNum(), 2.0)
            total_vdw_volume += (4/3) * np.pi * (radius ** 3)
    
    features['ffv_vdw_volume_per_repeat'] = total_vdw_volume
    
    # MW per volume (density indicator)
    if total_vdw_volume > 0:
        features['ffv_mw_per_vdw_volume'] = mw_per_unit / total_vdw_volume
    else:
        features['ffv_mw_per_vdw_volume'] = 0
    
    return features

def extract_all_features(smiles, include_3d=True, include_fp_arrays=True):
    """
    Extract ALL unique features: RDKit + custom polymer-specific + fingerprints + DP features
    
    Args:
        smiles: SMILES string
        include_3d: If True, generate 3D conformer and include 3D descriptors (11 features)
        include_fp_arrays: If True, include full fingerprint bit arrays (2239 features)
                          If False, only use derived fingerprint features (108 features)
    
    Returns:
        Dictionary with features:
        - Without 3D, without FP arrays: ~397 features
        - With 3D, without FP arrays: ~408 features
        - Without 3D, with FP arrays: ~2636 features
        - With 3D, with FP arrays: ~2647 features
        
        Breakdown (with 3D, with FP arrays):
        - 208 RDKit 2D
        - 2 RDKit Graph
        - 30 RDKit Molecular (19 + 11 3D)
        - 2347 Fingerprints (108 derived + 2239 raw bits)
        - 48 Custom polymer
        - 5 Degree of polymerization
        - 7 Packing & thermal
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    features = {}
    
    # 1. RDKit 2D descriptors (208)
    features.update(extract_all_rdkit_2d(mol))
    
    # 2. RDKit Graph descriptors (2 - only unique ones)
    features.update(extract_all_rdkit_graph(mol))
    
    # 3. RDKit Molecular descriptors (19 without 3D, 30 with 3D)
    features.update(extract_rdkit_molecular_descriptors(mol, include_3d=include_3d))
    
    # 4. Fingerprints: All 4 strategies + optional full arrays
    #    Without arrays: 108 features (counts, folded, hashed, statistical)
    #    With arrays: 2347 features (108 + 1024 Morgan + 1024 AtomPair + 1024 Torsion + 167 MACCS)
    features.update(calc_fingerprints_all_strategies(mol, include_full_arrays=include_fp_arrays))
    
    # 5. Custom polymer-specific features (48)
    features.update(extract_tc_positional_features(smiles))
    features.update(calc_backbone_sidechain(smiles))
    features.update(calc_graph_features(mol))
    features.update(calc_paper_based_features(smiles))
    features.update(calc_derived(features))
    
    # 6. Degree of polymerization features (5) - CRITICAL for MD-based properties
    features.update(calc_degree_of_polymerization(smiles))
    
    # 7. Packing & thermal features (7) - For density, FFV, Tg, thermal conductivity
    features.update(calc_packing_and_thermal_features(smiles))
    
    # 8. Final cleaning: Replace any remaining inf/nan/extreme values
    for key, value in features.items():
        if isinstance(value, float):
            if np.isnan(value):
                features[key] = None  # NaN → None (AutoGluon handles this)
            elif np.isinf(value):
                # Inf → large but finite value
                features[key] = 1e6 if value > 0 else -1e6
            elif abs(value) > 1e10:
                # Extreme values (like Ipc=1e59) → clip to reasonable range
                features[key] = 1e10 if value > 0 else -1e10
    
    return features
