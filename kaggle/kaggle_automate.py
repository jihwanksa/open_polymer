#!/usr/bin/env python3
"""
Kaggle Notebook Automation - Push, Execute, Submit, and Track Scores
Usage: python kaggle_automate.py [message]
"""

import subprocess
import time
import sys
import os
from datetime import datetime

KERNEL_SLUG = "jihwano/polymer-v2-enhanced-tc-tg-augmentation"
COMPETITION = "neurips-open-polymer-prediction-2025"
WORK_DIR = "/Users/jihwan/Downloads/open_polymer"

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

def run_cmd(cmd):
    """Run a shell command silently"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def log_print(msg, log_file=None):
    """Print to console and log file"""
    print(msg)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

def main():
    log_file = os.path.join(WORK_DIR, "kaggle", "automation.log")
    
    # Clear previous log
    with open(log_file, 'w') as f:
        f.write(f"=== Automation started at {datetime.now()} ===\n")
    
    log_print(f"\n{'='*70}", log_file)
    log_print(f"🚀 KAGGLE AUTOMATION PIPELINE", log_file)
    log_print(f"{'='*70}\n", log_file)
    start = time.time()
    
    # Step 1: Push
    log_print(f"[{timestamp()}] Pushing notebook to Kaggle...", log_file)
    os.chdir(WORK_DIR)
    code, out, err = run_cmd("kaggle kernels push -p . 2>&1")
    if code == 0 and "successfully pushed" in out:
        log_print(f"✅ Pushed successfully\n", log_file)
    else:
        log_print(f"❌ Push failed: {err}\n", log_file)
        return
    
    # Step 2: Wait for execution
    log_print(f"[{timestamp()}] Waiting for notebook execution...", log_file)
    start_exec = time.time()
    check = 0
    while time.time() - start_exec < 1200:  # 20 min timeout
        check += 1
        code, out, _ = run_cmd(f"kaggle kernels status {KERNEL_SLUG}")
        elapsed = int(time.time() - start_exec)
        
        if "COMPLETE" in out:
            log_print(f"✅ Complete after {elapsed}s\n", log_file)
            break
        elif "ERROR" in out or "FAILED" in out:
            log_print(f"❌ Execution failed\n", log_file)
            return
        
        if check % 12 == 1:  # Print every 60 seconds
            log_print(f"   ⏳ Running... ({elapsed}s)", log_file)
        time.sleep(5)
    
    # Step 3: Download (downloads both output and logs)
    log_print(f"[{timestamp()}] Downloading results...", log_file)
    code, _, _ = run_cmd(f"kaggle kernels output {KERNEL_SLUG} -p .")
    if code == 0:
        log_print(f"✅ Downloaded\n", log_file)
        
        # Move log to logs folder with timestamp
        kernel_name = KERNEL_SLUG.split('/')[-1]
        original_log = os.path.join(WORK_DIR, f"{kernel_name}.log")
        if os.path.exists(original_log):
            logs_dir = os.path.join(WORK_DIR, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_log = os.path.join(logs_dir, f"{kernel_name}_{timestamp_str}.log")
            os.rename(original_log, versioned_log)
            log_print(f"📋 Notebook log saved: logs/{os.path.basename(versioned_log)}", log_file)
        else:
            log_print(f"⚠ Notebook log not found", log_file)
    else:
        log_print(f"❌ Download failed\n", log_file)
        return
    
    # Step 4: Submit
    log_print(f"[{timestamp()}] Submitting to competition...", log_file)
    submissions_dir = os.path.join(WORK_DIR, 'submissions')
    if os.path.exists(submissions_dir):
        files = sorted([f for f in os.listdir(submissions_dir) if f.endswith('.csv')], reverse=True)
        submission_file = os.path.join(submissions_dir, files[0]) if files else None
    else:
        submission_file = "submission.csv" if os.path.exists("submission.csv") else None
    
    if not submission_file:
        log_print(f"❌ No submission file found\n", log_file)
        return
    
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else f"Auto {datetime.now().strftime('%H:%M')}"
    code, _, _ = run_cmd(f'kaggle competitions submit -c {COMPETITION} -f {submission_file} -m "{msg}" 2>&1')
    log_print(f"✅ Submitted\n", log_file)
    
    # Step 5: Wait for scores
    log_print(f"[{timestamp()}] Waiting for scores...", log_file)
    for i in range(120):
        code, out, _ = run_cmd(f"kaggle competitions submissions -c {COMPETITION} --csv")
        if code == 0 and out:
            lines = out.strip().split('\n')
            if len(lines) >= 2 and "PENDING" not in lines[1]:
                log_print(f"✅ Scores received!\n", log_file)
                log_print(lines[0], log_file)
                log_print(lines[1], log_file)
                if len(lines) > 2:
                    log_print("\nPrevious:", log_file)
                    for prev in lines[2:min(5, len(lines))]:
                        log_print(prev, log_file)
                break
        
        if i % 6 == 0:  # Print every 60 seconds
            log_print(f"   ⏳ Waiting... ({i*10}s)", log_file)
        time.sleep(10)
    
    total = int(time.time() - start)
    log_print(f"\n{'='*70}", log_file)
    log_print(f"✅ DONE! Total time: {total//60}m {total%60}s", log_file)
    log_print(f"{'='*70}\n", log_file)

if __name__ == "__main__":
    main()
