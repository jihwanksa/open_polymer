"""
Open Polymer - Interactive Web Demo
AI-Powered Polymer Property Prediction

Author: Jihwan Oh
Date: October 2025
"""

import gradio as gr
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, 'src'))

import pickle

# Load v53 Best Random Forest Model
MODEL_LOADED = False
model = None

try:
    model_path = os.path.join(parent_dir, 'models', 'random_forest_v53_best.pkl')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data
    MODEL_LOADED = True
    print(f"✅ Loaded v53 Best Random Forest Model (Private: 0.07874, Public: 0.10354)")
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠️  Warning: Could not load model: {e}")

# Property descriptions
PROPERTY_INFO = {
    'Tg': {
        'name': 'Glass Transition Temperature (Tg)',
        'unit': '°C',
        'description': 'Temperature where polymer transitions from hard/glassy to soft/rubbery state',
        'typical_range': '-100 to 300°C',
        'applications': 'Thermal stability, processing conditions'
    },
    'FFV': {
        'name': 'Fractional Free Volume (FFV)',
        'unit': 'dimensionless',
        'description': 'Fraction of unoccupied space in polymer matrix',
        'typical_range': '0.01 to 0.30',
        'applications': 'Gas permeability, membrane separation'
    },
    'Tc': {
        'name': 'Critical Temperature (Tc)',
        'unit': 'K',
        'description': 'Temperature above which gas cannot be liquefied',
        'typical_range': '200 to 1000 K',
        'applications': 'Phase behavior, processing windows'
    },
    'Density': {
        'name': 'Density',
        'unit': 'g/cm³',
        'description': 'Mass per unit volume of polymer',
        'typical_range': '0.8 to 2.0 g/cm³',
        'applications': 'Weight calculations, mechanical properties'
    },
    'Rg': {
        'name': 'Radius of Gyration (Rg)',
        'unit': 'Å',
        'description': 'Average distance of polymer chain from center of mass',
        'typical_range': '5 to 50 Å',
        'applications': 'Molecular size, chain conformation'
    }
}

def create_chemistry_features_single(smiles):
    """Create 21 chemistry-based features from a single SMILES string (v53 configuration)"""
    try:
        smiles_str = str(smiles) if pd.notna(smiles) else ""
        
        # Basic counts (10 features)
        basic = {
            'smiles_length': len(smiles_str),
            'carbon_count': smiles_str.count('C'),
            'nitrogen_count': smiles_str.count('N'),
            'oxygen_count': smiles_str.count('O'),
            'sulfur_count': smiles_str.count('S'),
            'fluorine_count': smiles_str.count('F'),
            'ring_count': smiles_str.count('c') + smiles_str.count('C1'),
            'double_bond_count': smiles_str.count('='),
            'triple_bond_count': smiles_str.count('#'),
            'branch_count': smiles_str.count('('),
        }
        
        # Chemistry-based features (11 additional features)
        num_side_chains = smiles_str.count('(')
        backbone_carbons = smiles_str.count('C') - smiles_str.count('C(')
        aromatic_count = smiles_str.count('c')
        h_bond_donors = smiles_str.count('O') + smiles_str.count('N')
        h_bond_acceptors = smiles_str.count('O') + smiles_str.count('N')
        num_rings = smiles_str.count('1') + smiles_str.count('2')
        single_bonds = len(smiles_str) - smiles_str.count('=') - smiles_str.count('#') - aromatic_count
        halogen_count = smiles_str.count('F') + smiles_str.count('Cl') + smiles_str.count('Br')
        heteroatom_count = smiles_str.count('N') + smiles_str.count('O') + smiles_str.count('S')
        mw_estimate = (smiles_str.count('C') * 12 + smiles_str.count('O') * 16 + 
                      smiles_str.count('N') * 14 + smiles_str.count('S') * 32 + smiles_str.count('F') * 19)
        branching_ratio = num_side_chains / max(backbone_carbons, 1)
        
        # Combine all features (21 total)
        features = {
            **basic,
            'num_side_chains': num_side_chains,
            'backbone_carbons': backbone_carbons,
            'aromatic_count': aromatic_count,
            'h_bond_donors': h_bond_donors,
            'h_bond_acceptors': h_bond_acceptors,
            'num_rings': num_rings,
            'single_bonds': single_bonds,
            'halogen_count': halogen_count,
            'heteroatom_count': heteroatom_count,
            'mw_estimate': mw_estimate,
            'branching_ratio': branching_ratio,
        }
        
        return np.array(list(features.values())).reshape(1, -1)
    except:
        return np.zeros((1, 21))


def predict_properties(smiles_input):
    """Main prediction function"""
    
    if not MODEL_LOADED:
        return (
            None,
            "❌ Error: Model not loaded. Please run training first.",
            "",
            ""
        )
    
    # Validate SMILES
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        return (
            None,
            "❌ Invalid SMILES string. Please check your input.",
            "",
            ""
        )
    
    try:
        # Draw molecule
        img = Draw.MolToImage(mol, size=(400, 400))
        
        # Extract chemistry features (21 features - v53 configuration)
        X = create_chemistry_features_single(smiles_input)
        
        # Predict using v53 ensemble model
        target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        predictions_raw = np.zeros(5)
        
        for i, target in enumerate(target_cols):
            if target in model['models'] and target in model['scalers']:
                scaler = model['scalers'][target]
                ensemble_models = model['models'][target]
                
                X_scaled = scaler.transform(X)
                ensemble_preds = np.array([m.predict(X_scaled) for m in ensemble_models])
                predictions_raw[i] = ensemble_preds.mean()
        
        # Apply Tg transformation (v53: (9/5) * Tg + 45)
        predictions = predictions_raw.copy()
        predictions[0] = (9/5) * predictions[0] + 45  # Tg transformation
        
        # Format results
        results_md = "## 🎯 Predicted Properties\n\n"
        results_md += "| Property | Value | Unit | Description |\n"
        results_md += "|----------|-------|------|-------------|\n"
        
        prop_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        for i, prop_name in enumerate(prop_names):
            info = PROPERTY_INFO[prop_name]
            value = predictions[i]
            results_md += f"| **{info['name']}** | **{value:.3f}** | {info['unit']} | {info['description']} |\n"
        
        # Add confidence note
        results_md += "\n### 📊 Model Performance\n"
        results_md += "- **Model:** Random Forest Ensemble (v53 - Best 🥇)\n"
        results_md += "- **Private Score:** 0.07874 | **Public Score:** 0.10354\n"
        results_md += "- **Training Data:** 10,039 samples (with external data augmentation)\n"
        results_md += "- **Ensemble:** 5 models per property\n"
        results_md += "- **Validation:** NeurIPS 2025 Competition\n"
        
        # Molecular descriptors
        desc_md = "## 🧪 Molecular Descriptors\n\n"
        desc_md += "| Descriptor | Value | Meaning |\n"
        desc_md += "|------------|-------|----------|\n"
        desc_md += f"| **Molecular Weight** | {Descriptors.MolWt(mol):.2f} g/mol | Size of molecule |\n"
        desc_md += f"| **LogP** | {Descriptors.MolLogP(mol):.2f} | Lipophilicity (fat solubility) |\n"
        desc_md += f"| **TPSA** | {Descriptors.TPSA(mol):.2f} Ų | Polar surface area |\n"
        desc_md += f"| **Rotatable Bonds** | {Descriptors.NumRotatableBonds(mol)} | Chain flexibility |\n"
        desc_md += f"| **H-Bond Donors** | {Descriptors.NumHDonors(mol)} | Hydrogen bonding |\n"
        desc_md += f"| **H-Bond Acceptors** | {Descriptors.NumHAcceptors(mol)} | Hydrogen bonding |\n"
        desc_md += f"| **Aromatic Rings** | {Descriptors.NumAromaticRings(mol)} | Rigidity |\n"
        
        # Application guidance
        guidance_md = "## 💡 Application Guidance\n\n"
        
        # Analyze predictions
        tg, ffv, tc, density, rg = predictions
        
        guidance_md += "### Material Characteristics:\n\n"
        
        # Note: Tg is already transformed, so use the displayed value
        if tg > 200:
            guidance_md += "- ✅ **High Tg ({:.1f}°C transformed)**: Good thermal stability, suitable for high-temp applications\n".format(tg)
        elif tg > 100:
            guidance_md += "- ⚠️ **Medium Tg ({:.1f}°C transformed)**: Moderate thermal stability\n".format(tg)
        else:
            guidance_md += "- ❄️ **Low Tg ({:.1f}°C transformed)**: Flexible at room temperature, good for elastomers\n".format(tg)
        
        if ffv > 0.15:
            guidance_md += "- 🌬️ **High FFV ({:.3f})**: Excellent for gas separation membranes\n".format(ffv)
        elif ffv > 0.10:
            guidance_md += "- ✅ **Medium FFV ({:.3f})**: Good permeability\n".format(ffv)
        else:
            guidance_md += "- 🔒 **Low FFV ({:.3f})**: Dense material, good barrier properties\n".format(ffv)
        
        if density > 1.5:
            guidance_md += "- ⚖️ **High Density ({:.2f} g/cm³)**: Heavy, strong material\n".format(density)
        elif density > 1.0:
            guidance_md += "- ⚖️ **Medium Density ({:.2f} g/cm³)**: Standard polymer\n".format(density)
        else:
            guidance_md += "- 🪶 **Low Density ({:.2f} g/cm³)**: Lightweight material\n".format(density)
        
        guidance_md += "\n### Recommended Applications:\n\n"
        
        # Application recommendations
        apps = []
        if tg > 200:
            apps.append("- 🔥 High-temperature engineering plastics")
            apps.append("- 🏗️ Structural components")
        if ffv > 0.15:
            apps.append("- 🧊 Gas separation membranes")
            apps.append("- 💨 Breathable films")
        if density < 1.0:
            apps.append("- ✈️ Lightweight composites")
            apps.append("- 🏃 Sports equipment")
        if 0.05 < ffv < 0.15:
            apps.append("- 📦 Packaging materials")
            apps.append("- 🧴 Consumer products")
        
        if apps:
            guidance_md += "\n".join(apps)
        else:
            guidance_md += "- 🔬 Specialty applications (consult materials engineer)"
        
        return (
            img,
            results_md,
            desc_md,
            guidance_md
        )
        
    except Exception as e:
        return (
            None,
            f"❌ Error during prediction: {str(e)}",
            "",
            ""
        )

# Example molecules
EXAMPLES = [
    ["*CC(*)CCCC", "Polyethylene-like polymer"],
    ["*c1ccccc1*", "Polystyrene-like polymer"],
    ["*CC(*)C(=O)OC", "Poly(methyl methacrylate)-like"],
    ["*CC(*)C#N", "Polyacrylonitrile-like"],
    ["*C(*)CF", "Poly(vinyl fluoride)-like"]
]

# Create Gradio interface
with gr.Blocks(title="Open Polymer - AI Property Prediction", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🧪 Open Polymer: AI-Powered Property Prediction
    
    ## Instant polymer property predictions from molecular structure
    
    [![GitHub](https://img.shields.io/badge/GitHub-open__polymer-blue?logo=github)](https://github.com/jihwanksa/open_polymer)
    [![Accuracy](https://img.shields.io/badge/Accuracy-97%25-success)](https://github.com/jihwanksa/open_polymer)
    [![Model](https://img.shields.io/badge/Model-XGBoost-orange)](https://github.com/jihwanksa/open_polymer)
    
    **Predict 5 key polymer properties in under 1 second!**
    
    Enter a SMILES string representing a polymer repeat unit. The asterisks (*) denote connection points.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            smiles_input = gr.Textbox(
                label="SMILES Input",
                placeholder="*CC(*)CCCC",
                info="Enter polymer SMILES (use * for connection points)",
                lines=2
            )
            
            predict_btn = gr.Button("🚀 Predict Properties", variant="primary", size="lg")
            
            gr.Markdown("### 📝 Example Polymers")
            for smiles, name in EXAMPLES:
                gr.Button(f"{name}", size="sm").click(
                    lambda s=smiles: s,
                    outputs=smiles_input
                )
            
            gr.Markdown("""
            ### ℹ️ About This Model
            
            - **Model:** Random Forest v53 🥇
            - **Score:** 0.07874 (Private)
            - **Ensemble:** 5 models/property
            - **Speed:** < 1 second
            - **Training:** 10,039 polymers
            - **Validation:** NeurIPS 2025
            
            ### 🎯 Properties Predicted
            
            1. **Tg** - Glass Transition Temp
            2. **FFV** - Fractional Free Volume
            3. **Tc** - Critical Temperature
            4. **Density** - Material density
            5. **Rg** - Radius of Gyration
            """)
        
        with gr.Column(scale=2):
            with gr.Tab("Visualization"):
                molecule_img = gr.Image(label="Molecular Structure", type="pil")
            
            with gr.Tab("Predictions"):
                results_md = gr.Markdown()
            
            with gr.Tab("Molecular Descriptors"):
                descriptors_md = gr.Markdown()
            
            with gr.Tab("Application Guidance"):
                guidance_md = gr.Markdown()
    
    gr.Markdown("""
    ---
    
    ## 🚀 How It Works
    
    1. **Input:** Enter SMILES string of polymer repeat unit
    2. **Feature Extraction:** Compute 21 chemistry-based features (simple, effective!)
    3. **Prediction:** Random Forest Ensemble (5 models per property) predicts 5 properties
    4. **Transformation:** Apply Tg shift correction (competition discovery)
    5. **Visualization:** Display molecule structure and results
    
    ## 📊 Model Performance (v53 - Best)
    
    | Property | R² Score | MAE | Training Samples |
    |----------|----------|-----|------------------|
    | FFV | 0.750 ⭐⭐ | 0.0096 | 7,030 samples |
    | Density | 0.678 ⭐ | 0.0410 g/cm³ | 613 samples |
    | Tg | 0.672 ⭐ | 46.55 °C | 2,447 samples |
    | Tc | 0.444 | 0.0361 K | 867 samples |
    | Rg | 0.434 | 2.50 Å | 614 samples |
    
    **Competition Score:** Private 0.07874, Public 0.10354 🥇
    
    ## 🔬 Technology Stack
    
    - **ML:** Random Forest Ensemble (5 models × 5 properties = 25 total models)
    - **Features:** 21 chemistry-based features (simple but effective!)
    - **Chemistry:** RDKit for SMILES parsing and visualization
    - **Interface:** Gradio for interactive web demo
    - **Deployment:** CPU-based (no GPU required)
    - **Training:** External data augmentation (Tc, Tg, Density, Rg)
    
    ## 📚 Learn More
    
    - **GitHub:** [github.com/jihwanksa/open_polymer](https://github.com/jihwanksa/open_polymer)
    - **Documentation:** See README.md in repository
    - **VC Pitch:** See docs/VC_PITCH.md for business case
    
    ## 📞 Contact
    
    **Jihwan Oh** | [@jihwanksa](https://github.com/jihwanksa)  
    Built with ❤️ using XGBoost, RDKit, and Gradio
    
    ---
    
    *Disclaimer: Predictions are for research purposes. Always validate critical applications with experimental testing.*
    """)
    
    # Connect button to function
    predict_btn.click(
        fn=predict_properties,
        inputs=smiles_input,
        outputs=[molecule_img, results_md, descriptors_md, guidance_md]
    )
    
    # Also trigger on Enter key
    smiles_input.submit(
        fn=predict_properties,
        inputs=smiles_input,
        outputs=[molecule_img, results_md, descriptors_md, guidance_md]
    )

# Launch instructions
if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Open Polymer - Interactive Demo")
    print("=" * 80)
    print()
    print("Starting Gradio interface...")
    print()
    
    if not MODEL_LOADED:
        print("⚠️  WARNING: Model not loaded!")
        print("Please run training first:")
        print("  python src/train.py")
        print()
    else:
        print("✅ Model loaded successfully")
        print()
    
    print("Opening browser...")
    print("=" * 80)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )

