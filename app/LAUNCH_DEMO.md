# 🚀 Launch Interactive Demo

Quick guide to running the Open Polymer web interface.

## Prerequisites

Make sure you have:
1. ✅ Conda environment activated: `conda activate polymer_pred`
2. ✅ Gradio installed: `pip install gradio==4.10.0`
3. ✅ Trained models in `models/` directory

## Quick Launch

```bash
# From project root
python app.py
```

The app will automatically:
- Load the XGBoost model
- Start a local web server
- Open your browser to http://localhost:7861

## Usage

### 1. Enter SMILES String
Type or select a polymer SMILES in the input box. Use `*` for connection points:
```
*CC(*)CCCC     # Polyethylene-like
*c1ccccc1*     # Polystyrene-like
```

### 2. View Results
The interface shows 4 tabs:
- **Visualization**: 2D molecular structure
- **Predictions**: 5 predicted properties with accuracy info
- **Molecular Descriptors**: Chemical features
- **Application Guidance**: Suggested use cases

### 3. Example Polymers
Click any example button to instantly load and predict.

## Sharing the Demo

### Local Network
```bash
# Allow access from other devices on your network
python app.py
# Note the "Running on local URL" address
```

### Public URL (temporary)
```python
# In app.py, change:
demo.launch(share=True)  # Creates public ngrok link
```

⚠️ **Warning**: Public links expire after 72 hours and expose your demo to the internet.

## Customization

### Change Port
```python
# In app.py, modify:
demo.launch(server_port=8080)  # Use port 8080 instead of 7861
```

### Enable Authentication
```python
# In app.py, add:
demo.launch(auth=("username", "password"))
```

### Dark Mode
The interface uses Gradio's Soft theme. To change:
```python
# In app.py, modify:
with gr.Blocks(theme=gr.themes.Glass()) as demo:  # Different theme
```

## Troubleshooting

### "Model not loaded"
```bash
# Train the model first
python src/train.py

# Verify model exists
ls -lh models/xgboost_model.pkl
```

### "Port 7861 already in use"
```bash
# Kill existing process
lsof -ti:7861 | xargs kill -9

# Or use different port (see above)
```

### "RDKit import error"
```bash
# Reinstall RDKit
conda install -c conda-forge rdkit -y
```

### Slow predictions
- Check CPU usage
- Close other applications
- Model loads on first prediction (normal)

## Screenshots

### Main Interface
- Clean, professional design
- Molecule visualization with RDKit
- Color-coded results tabs

### Predictions Tab
- 5 properties with units
- Model accuracy information
- Training dataset details

### Application Guidance
- Smart recommendations based on properties
- Industry use cases
- Material characteristics

## API Mode

To use as an API instead of web interface:

```python
from app import predict_properties

# Predict programmatically
smiles = "*CC(*)CCCC"
img, results, desc, guidance = predict_properties(smiles)

print(results)  # Markdown-formatted results
```

## Deployment Options

### Production Deployment

#### Option 1: Gradio Spaces (Free)
```bash
# Create account at huggingface.co/spaces
# Upload: app.py, models/, src/
# Set requirements.txt
# Auto-deploys
```

#### Option 2: Docker
```bash
# Create Dockerfile
# Include: Python 3.10, requirements, models
# Run: docker run -p 7861:7861 polymer-app
```

#### Option 3: Cloud VM
```bash
# AWS/GCP/Azure VM
# Install dependencies
# Run with nohup: nohup python app.py &
# Configure firewall for port 7861
```

## Performance

- **Loading time**: 2-3 seconds (model load)
- **Prediction time**: < 100ms per molecule
- **Memory usage**: ~500MB (model + interface)
- **Concurrent users**: 10-50 (depends on CPU)

## Support

Issues? Check:
1. [README.md](README.md) - Main documentation
2. [src/README.md](src/README.md) - Code documentation
3. GitHub Issues - Report bugs

---

**Enjoy predicting polymer properties! 🧪**

