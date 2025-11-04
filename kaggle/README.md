# Kaggle Automation Workflow

Fully automated push, execute, submit, and score tracking for Kaggle competitions.

## Quick Start

```bash
cd /Users/jihwan/Downloads/open_polymer
python kaggle/kaggle_automate.py "Your message"
```

**Output files:**
- `kaggle/automation.log` - Workflow progress
- `logs/polymer-v2-enhanced-tc-tg-augmentation_YYYYMMDD_HHMMSS.log` - Notebook execution details
- `submission.csv` - Latest predictions

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
2. ✅ **Waits** for execution (polls every 60 seconds, max 20 minutes)
3. ✅ **Downloads** output files and notebook execution logs
4. ✅ **Submits** automatically to competition
5. ✅ **Polls** for scores (up to 20 minutes)
6. ✅ **Displays** results with history

## Output Organization

Automatically organizes all outputs:
- **Automation logs:** `kaggle/automation.log` (tracks automation workflow)
- **Notebook logs:** `logs/polymer-v2-enhanced-tc-tg-augmentation_YYYYMMDD_HHMMSS.log` (timestamped execution logs)
- **Submissions:** `submissions/submission_YYYYMMDD_HHMMSS.csv` (if created by notebook)
- **Latest submission:** `submission.csv` (current output)

## Example Output

```
======================================================================
🚀 KAGGLE AUTOMATION PIPELINE
======================================================================

[10:06:22] Pushing notebook to Kaggle...
✅ Pushed successfully

[10:06:24] Waiting for notebook execution...
   ⏳ Running... (0s)
✅ Complete after 53s

[10:07:18] Downloading results...
✅ Downloaded

📋 Notebook log saved: logs/polymer-v2-enhanced-tc-tg-augmentation_20251104_100718.log

[10:07:18] Submitting to competition...
✅ Submitted

[10:07:20] Waiting for scores...
✅ Scores received!

fileName,date,description,status,publicScore,privateScore
submission.csv,2025-10-31 18:10:28.907000,,SubmissionStatus.COMPLETE,0.11413,0.08334

Previous:
submission.csv,2025-10-31 17:52:35.073000,,SubmissionStatus.COMPLETE,0.10049,0.08548
submission.csv,2025-10-31 17:08:26.517000,,SubmissionStatus.COMPLETE,0.10049,0.08548

======================================================================
✅ DONE! Total time: 0m 57s
======================================================================
```

## Checking Execution Logs

Notebook execution logs are saved with timestamps in `logs/` folder:

```bash
# View latest log
ls -lt logs/ | head -2

# Check specific log
cat logs/polymer-v2-enhanced-tc-tg-augmentation_20251104_100718.log

# View automation workflow
cat kaggle/automation.log
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
- Notebook execution logs are saved with timestamps for version tracking
- Automation logs track the workflow progress (push → execute → submit → score)
- All submissions show current + previous 3 scores for comparison
- Perfect for iterative development and CI/CD pipeline integration

## Features

- ✅ **Automated workflow**: One command runs entire pipeline
- ✅ **Version tracking**: Timestamped logs for each execution
- ✅ **Score history**: Shows last 4 submissions automatically
- ✅ **Error handling**: Gracefully handles failures at each step
- ✅ **Progress monitoring**: Real-time status updates during execution
