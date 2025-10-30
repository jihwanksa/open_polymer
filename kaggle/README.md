# Kaggle Automation Workflow

Fully automated push, execute, submit, and score tracking for Kaggle competitions.

## Directory Structure

```
kaggle/
├── kaggle_automate.py      # Main automation script
├── kernel-metadata.json    # Kaggle notebook metadata
└── README.md              # This file
```

## Setup

1. **Install Kaggle CLI:**
   ```bash
   pip install kaggle
   ```

2. **Configure Kaggle API:**
   - Download `kaggle.json` from https://www.kaggle.com/settings/account
   - Place it at `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Usage

### Basic Usage (Auto-submit)
```bash
cd /Users/jihwan/Downloads/open_polymer
python kaggle/kaggle_automate.py "Your message here"
```

### Example Messages
```bash
# Test run
python kaggle/kaggle_automate.py "v14: testing new features"

# Version update
python kaggle/kaggle_automate.py "v15: Added Tg transformation"

# Default (auto-dated)
python kaggle/kaggle_automate.py
```

## What It Does

1. ✅ **Pushes** notebook to Kaggle
2. ✅ **Waits** for execution (polls every 15 seconds, max 10 minutes)
3. ✅ **Downloads** output files and logs
4. ✅ **Shows** last 50 lines of kernel logs
5. ✅ **Submits** automatically to competition
6. ✅ **Polls** for scores (up to 20 minutes)
7. ✅ **Displays** results with history

## Output Organization

Automatically organizes outputs:
- **Logs:** `../logs/polymer-v2-enhanced-tc-tg-augmentation.log`
- **Submissions:** `../submissions/submission_YYYYMMDD_HHMMSS.csv`

## Example Output

```
======================================================================
🚀 KAGGLE NOTEBOOK AUTOMATION
======================================================================

▶ Pushing notebook to Kaggle...
======================================================================
✅ Notebook pushed successfully!

⏳ Waiting for notebook execution...
  [125s] Still running...
✅ Notebook execution complete!

📥 Downloading notebook output...
  ✅ Output downloaded successfully!

📋 KERNEL LOG (last 50 lines):
======================================================================
[Training XGBoost for Tg...]
[Training XGBoost for FFV...]
...
✅ Submission saved to submission.csv

📤 Submitting to competition...
✅ Submission successful!

⏳ Waiting for Kaggle to score submission...
  [95s] Still scoring...

✅ SCORES RECEIVED!
======================================================================
SUBMISSION RESULTS:
======================================================================
fileName,date,description,status,publicScore,privateScore
-
submission.csv,2025-10-27 22:15:00,,COMPLETE,0.082,0.083
```

## Configuration

Edit variables in `kaggle_automate.py`:
```python
KERNEL_SLUG = "jihwano/polymer-v2-enhanced-tc-tg-augmentation"
COMPETITION = "neurips-open-polymer-prediction-2025"
WORK_DIR = "/Users/jihwan/Downloads/open_polymer"
```

## Troubleshooting

### "Notebook not found"
- Ensure `kernel-metadata.json` is in the open_polymer root directory
- Check kernel slug matches your Kaggle username/notebook

### "No submission.csv found"
- Notebook must create `submission.csv` in its output
- Check notebook execution logs

### "API error"
- Wait 1-2 minutes and try manual submission
- Check Kaggle API limits (daily submissions cap)
- Verify kaggle.json is properly configured

## Notes

- Each run is fully independent
- Submissions are timestamped and stored
- Logs preserve the complete execution history
- Perfect for CI/CD pipeline integration
