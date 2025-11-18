"""
Setup script for Google Colab environment

Run this in the first Colab cell to install dependencies and setup paths
"""

def setup_colab():
    """Setup Google Colab environment"""
    import subprocess
    import sys
    
    print("🚀 Setting up Google Colab environment...")
    print("\n1️⃣ Mounting Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    
    print("\n2️⃣ Installing dependencies...")
    # Install required packages
    packages = [
        'pandas',
        'numpy',
        'rdkit',
        'scikit-learn',
        'tqdm',
        'autogluon',
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    
    print("\n3️⃣ Verifying GPU...")
    import torch
    print(f"   GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Upload your project to Google Drive or download from GitHub")
    print("2. Update the PROJECT_ROOT variable in the training script")
    print("3. Run the training script")

if __name__ == "__main__":
    setup_colab()
