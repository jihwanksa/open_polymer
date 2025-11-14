"""
Download and setup pre-trained models from Hugging Face for pseudo-label generation.

This script downloads the three models used in the 1st place solution:
1. BERT-base-SMILES (from unikei/bert-base-smiles)
2. Uni-Mol 2 (from dptech/Uni-Mol2)
3. AutoGluon Tabular (from Hugging Face model hub)

Models will be saved in the models/ directory for use in pseudo-label generation.
"""

import os
import sys
import argparse
from pathlib import Path

def setup_bert_smiles(model_dir: str = "models/bert_smiles"):
    """Download BERT-base-SMILES from Hugging Face"""
    print("\n" + "="*80)
    print("1️⃣  SETTING UP BERT-base-SMILES")
    print("="*80)
    
    try:
        from transformers import AutoTokenizer, AutoModelForPreTraining
        
        os.makedirs(model_dir, exist_ok=True)
        
        print("\n📥 Downloading BERT tokenizer from Hugging Face...")
        print("   Model: unikei/bert-base-smiles")
        
        tokenizer = AutoTokenizer.from_pretrained("unikei/bert-base-smiles")
        tokenizer.save_pretrained(model_dir)
        print(f"   ✅ Tokenizer saved to {model_dir}/")
        
        print("\n📥 Downloading BERT model from Hugging Face...")
        model = AutoModelForPreTraining.from_pretrained("unikei/bert-base-smiles")
        model.save_pretrained(model_dir)
        print(f"   ✅ Model saved to {model_dir}/")
        
        print(f"\n✅ BERT-base-SMILES setup complete!")
        print(f"   Model path: {model_dir}")
        print(f"   Usage: Load with AutoModel.from_pretrained('{model_dir}')")
        
        return True
        
    except ImportError:
        print("❌ transformers library not installed!")
        print("   Install with: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ Failed to setup BERT-base-SMILES: {e}")
        return False


def setup_unimol2(model_dir: str = "models/unimol2"):
    """Download Uni-Mol 2 84M model from Hugging Face"""
    print("\n" + "="*80)
    print("2️⃣  SETTING UP UNI-MOL 2 (84M)")
    print("="*80)
    
    try:
        from huggingface_hub import hf_hub_download
        
        os.makedirs(model_dir, exist_ok=True)
        
        print("\n📥 Downloading Uni-Mol 2 model from Hugging Face...")
        print("   Repository: dptech/Uni-Mol2")
        print("   Model: 84M variant (checkpoint.pt)")
        
        # Download the 84M checkpoint
        model_path = hf_hub_download(
            repo_id="dptech/Uni-Mol2",
            filename="modelzoo/84M/checkpoint.pt",
            cache_dir=model_dir,
            force_download=False
        )
        
        print(f"   ✅ Model downloaded to: {model_path}")
        
        print(f"\n✅ Uni-Mol 2 setup complete!")
        print(f"   Model path: {model_path}")
        print(f"   Note: Uni-Mol requires specific installation. See README for details.")
        
        return True
        
    except ImportError:
        print("❌ huggingface_hub library not installed!")
        print("   Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Failed to setup Uni-Mol 2: {e}")
        print("   Troubleshooting:")
        print("   - Check internet connection")
        print("   - Verify Hugging Face model exists at dptech/Uni-Mol2")
        print("   - Try manual download from: https://huggingface.co/dptech/Uni-Mol2")
        return False


def setup_autogluon_model(model_dir: str = "models/autogluon"):
    """
    Setup AutoGluon model for tabular property prediction.
    
    NOTE: AutoGluon models on Hugging Face are typically dataset-specific.
    We recommend one of these options:
    1. Train your own AutoGluon model on the polymer dataset
    2. Use the pre-trained model from the 1st place solution (if available)
    3. Use a general chemistry property prediction model
    """
    print("\n" + "="*80)
    print("3️⃣  SETTING UP AUTOGLUON TABULAR MODEL")
    print("="*80)
    
    print("\n⚠️  AutoGluon Setup Notes:")
    print("   AutoGluon models on Hugging Face vary by task and dataset.")
    print("   Here are the recommended options:")
    print()
    print("   OPTION A: Train AutoGluon on polymer data (Recommended)")
    print("   ───────────────────────────────────────────────────")
    print("   from autogluon.tabular import TabularPredictor")
    print("   predictor = TabularPredictor(label='target_property')")
    print("   predictor.fit(train_data=df)")
    print("   predictor.save_space()")
    print()
    print("   OPTION B: Download pre-trained model from Hugging Face")
    print("   ────────────────────────────────────────────────────")
    print("   from huggingface_hub import hf_hub_download")
    print("   model_path = hf_hub_download(")
    print("       repo_id='<username>/<model-name>',")
    print("       filename='model.pkl'")
    print("   )")
    print()
    print("   OPTION C: Use ensemble without AutoGluon")
    print("   ──────────────────────────────────────────")
    print("   Use only BERT + Uni-Mol for pseudo-label generation")
    print()
    
    return None


def list_available_autogluon_models():
    """List some popular AutoGluon models on Hugging Face"""
    print("\n" + "="*80)
    print("AVAILABLE AUTOGLUON MODELS ON HUGGING FACE")
    print("="*80)
    
    models = [
        {
            "name": "AutoGluon Tabular (General)",
            "repo": "aurelien-git/autogluon-binary-classification",
            "description": "General binary classification model",
            "use_case": "Reference implementation"
        },
        {
            "name": "Chemistry Property Prediction",
            "repo": "search Hugging Face for 'chemistry property'",
            "description": "Various chemistry models",
            "use_case": "Domain-specific models"
        },
        {
            "name": "Polymer Property (if available)",
            "repo": "Search Hugging Face for 'polymer'",
            "description": "Polymer-specific models",
            "use_case": "Best match for this task"
        }
    ]
    
    print("\n📋 Popular AutoGluon Models:")
    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   Repo: {model['repo']}")
        print(f"   Description: {model['description']}")
        print(f"   Use case: {model['use_case']}")
    
    print("\n🔗 Search on Hugging Face:")
    print("   https://huggingface.co/models?search=autogluon")
    print("   https://huggingface.co/models?search=polymer+property")
    print("   https://huggingface.co/models?search=chemistry+property")


def main(args):
    print("\n" + "="*80)
    print("SETUP PRE-TRAINED MODELS FOR PSEUDO-LABEL GENERATION")
    print("="*80)
    print("\nThis script downloads models for the 1st place solution approach:")
    print("1. BERT-base-SMILES (SMILES encoding)")
    print("2. Uni-Mol 2 (molecular GNN)")
    print("3. AutoGluon (tabular predictions)")
    
    results = {}
    
    # Setup BERT-base-SMILES
    if args.bert or args.all:
        results['bert'] = setup_bert_smiles(args.bert_dir)
    
    # Setup Uni-Mol 2
    if args.unimol or args.all:
        results['unimol'] = setup_unimol2(args.unimol_dir)
    
    # Setup AutoGluon
    if args.autogluon or args.all:
        if args.list_autogluon:
            list_available_autogluon_models()
        setup_autogluon_model(args.autogluon_dir)
    
    # Summary
    print("\n" + "="*80)
    print("SETUP SUMMARY")
    print("="*80)
    
    if results:
        print("\n✅ Successfully downloaded:")
        for model_name, status in results.items():
            if status:
                print(f"   ✅ {model_name}")
            else:
                print(f"   ❌ {model_name} (check error above)")
    
    print("\n📝 Next Steps:")
    print("   1. Review models in the models/ directory")
    print("   2. Configure pseudo-label generation:")
    print("      python pseudolabel/generate_pseudolabels.py \\")
    print("          --input_data data/PI1M_50000_v2.1.csv \\")
    print("          --bert_model models/bert_smiles \\")
    print("          --unimol_model models/unimol2 \\")
    print("          --output_path pseudolabel/pi1m_pseudolabels.csv")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and setup pre-trained models from Hugging Face",
        epilog="""
Examples:

  1. Download all models:
     python pseudolabel/setup_pretrained_models.py --all
  
  2. Download only BERT and Uni-Mol:
     python pseudolabel/setup_pretrained_models.py --bert --unimol
  
  3. List available AutoGluon models:
     python pseudolabel/setup_pretrained_models.py --list-autogluon
  
  4. Custom output directory:
     python pseudolabel/setup_pretrained_models.py --all --bert-dir /custom/path/bert
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Download all models (BERT + Uni-Mol + AutoGluon info)"
    )
    parser.add_argument(
        "--bert",
        action="store_true",
        default=False,
        help="Download BERT-base-SMILES model"
    )
    parser.add_argument(
        "--unimol",
        action="store_true",
        default=False,
        help="Download Uni-Mol 2 model"
    )
    parser.add_argument(
        "--autogluon",
        action="store_true",
        default=False,
        help="Setup AutoGluon (shows instructions)"
    )
    parser.add_argument(
        "--list-autogluon",
        action="store_true",
        default=False,
        help="List available AutoGluon models on Hugging Face"
    )
    parser.add_argument(
        "--bert-dir",
        type=str,
        default="models/bert_smiles",
        help="Output directory for BERT model (default: models/bert_smiles)"
    )
    parser.add_argument(
        "--unimol-dir",
        type=str,
        default="models/unimol2",
        help="Output directory for Uni-Mol model (default: models/unimol2)"
    )
    parser.add_argument(
        "--autogluon-dir",
        type=str,
        default="models/autogluon",
        help="Output directory for AutoGluon (default: models/autogluon)"
    )
    
    args = parser.parse_args()
    
    # Default to showing help if no args
    if not any([args.all, args.bert, args.unimol, args.autogluon, args.list_autogluon]):
        parser.print_help()
        sys.exit(0)
    
    main(args)

