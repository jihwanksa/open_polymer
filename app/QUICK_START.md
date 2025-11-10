# Quick Start - Run the App

## ✅ Model Already Trained!
The v53 best Random Forest model has been trained and is ready to use.

## 🚀 Launch the App

### Step 1: Install Dependencies (if needed)
```bash
pip install gradio rdkit pandas numpy scikit-learn
```

### Step 2: Run the App
```bash
cd /Users/jihwan/Downloads/open_polymer/app
python app.py
```

### Step 3: Access the App
The app will launch automatically in your browser at:
- **http://localhost:7861**

## 🎯 Try It Out!

### Example SMILES to test:
1. `*CC(*)CCCC` - Polyethylene-like
2. `*c1ccccc1*` - Polystyrene-like  
3. `*CC(*)C(=O)OC` - PMMA-like
4. `*CC(*)C#N` - Polyacrylonitrile-like
5. `*C(*)CF` - Poly(vinyl fluoride)-like

## 📊 What You'll See

- **Molecular Structure** visualization
- **5 Property Predictions**:
  - Tg (Glass Transition Temperature)
  - FFV (Fractional Free Volume)
  - Tc (Critical Temperature)
  - Density
  - Rg (Radius of Gyration)
- **Molecular Descriptors** (MW, LogP, etc.)
- **Application Guidance** based on properties

## 🎉 Model Info

- **Version:** v53 - Best performing model
- **Type:** Random Forest Ensemble (5 models per property)
- **Features:** 21 chemistry-based features
- **Competition Score:**
  - Private: 0.07874 🥇
  - Public: 0.10354

## ⚠️ If Model Not Found

If you see "Model not loaded", run the training script:

```bash
cd /Users/jihwan/Downloads/open_polymer
python src/train_v53_best.py
```

This will take ~2-3 minutes and generate the model file.

---

**Enjoy predicting polymer properties! 🧪**

