# Colab Notebook for Systematic Feature Analysis

Copy-paste these cells into a Google Colab notebook to run the systematic feature analysis with GPU/TPU acceleration.

---

## Cell 1: Clone and Setup

```python
# Clone the project
!git clone https://github.com/jihwanksa/open_polymer.git
%cd open_polymer

# Install dependencies
!pip install -q autogluon rdkit pandas numpy tqdm scikit-learn

print("✅ Setup complete!")
```

---

## Cell 2: Train Single Configuration

Replace `C` with `A`, `B`, `D`, `E`, `F`, `G`, or `H` to test different configurations.

```python
import subprocess
import os

os.chdir('/content/open_polymer')

# Train configuration C (current baseline - 34 features)
result = subprocess.run([
    'python',
    'AutoGluon/systematic_feature/train_for_colab.py',
    '--config', 'C',
    '--time_limit', '300'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
```

---

## Cell 3: Train All Configurations (A-H)

Run all 8 configurations. **Estimated time: 30-40 minutes on T4 GPU**

```python
import subprocess
import os

os.chdir('/content/open_polymer')

# Train all configurations
result = subprocess.run([
    'python',
    'AutoGluon/systematic_feature/train_for_colab.py',
    '--all',
    '--time_limit', '300'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
```

---

## Cell 4: Check Results

```python
import json
import os
from pathlib import Path

results_dir = Path('/content/autogluon_results')

print("Configurations trained:")
for config_dir in sorted(results_dir.iterdir()):
    if config_dir.is_dir():
        results_json = config_dir / 'config_results.json'
        if results_json.exists():
            with open(results_json) as f:
                data = json.load(f)
            print(f"\n{data['config']}: {data['description']}")
            print(f"  Input features: {data['input_features']}")
            print(f"  Models trained: {len(data['models'])}")
            for target, model_info in data['models'].items():
                print(f"    - {target}: {model_info['selected_features']} features selected")
```

---

## Cell 5: Download Results

```python
from google.colab import files
import os
import shutil

# Create a zip file with all results
os.chdir('/content')
shutil.make_archive('autogluon_results', 'zip', '/content/autogluon_results')

# Download
files.download('autogluon_results.zip')

print("✅ Results downloaded!")
```

---

## Configuration Options

**Single Configuration:**
```bash
python AutoGluon/systematic_feature/train_for_colab.py --config A --time_limit 300
```

**Available configurations:**
- `A`: Simple only (10 features)
- `B`: Hand-crafted only (11 features)
- `C`: Current baseline (34 features) ← **Start here**
- `D`: Expanded RDKit (56 features)
- `E`: All RDKit (~81 features)
- `F`: RDKit only (35 features)
- `G`: No simple features (24 features)
- `H`: No hand-crafted features (23 features)

**All configurations:**
```bash
python AutoGluon/systematic_feature/train_for_colab.py --all --time_limit 300
```

---

## Expected Output

Each configuration will produce:
- `/content/autogluon_results/{CONFIG_NAME}/`
  - `Tg/` - AutoGluon model for Tg
  - `FFV/` - AutoGluon model for FFV
  - `Tc/` - AutoGluon model for Tc
  - `Density/` - AutoGluon model for Density
  - `Rg/` - AutoGluon model for Rg
  - `config_results.json` - Summary of training

---

## Timing Expectations (T4 GPU)

| Configuration | Features | Time |
|---|---|---|
| A (Simple) | 10 | 2-3 min |
| B (Hand-crafted) | 11 | 2-3 min |
| C (Current) | 34 | 3-4 min |
| D (Expanded RDKit) | 56 | 4-5 min |
| E (All RDKit) | ~81 | 5-7 min |
| F (RDKit only) | 35 | 3-4 min |
| G (No simple) | 24 | 3-4 min |
| H (No hand-crafted) | 23 | 3-4 min |
| **All (A-H)** | - | **30-40 min** |

---

## Troubleshooting

### "Project not found"
The script automatically checks these paths:
- `/content/open_polymer`
- `/content/drive/MyDrive/open_polymer`
- `/root/open_polymer`

If not found, you may need to upload manually or adjust the path in the script.

### "No module named 'autogluon'"
Run Cell 1 again to install dependencies.

### Out of memory
- Reduce `--time_limit` to 180 or 120
- Train configurations one at a time instead of `--all`

### Slow training
Check if GPU is enabled:
- Runtime → Change runtime type → GPU (T4)
- Monitor in Runtime → Show Resources

---

## Next Steps

1. Run Cell 1 (setup)
2. Run Cell 2 or 3 to train
3. Run Cell 4 to check results
4. Run Cell 5 to download
5. Analyze results locally

Good luck! 🚀

