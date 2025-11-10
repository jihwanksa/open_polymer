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

def fetch_logs(kernel_slug, work_dir):
    """Fetch and display execution logs"""
    log_dir = os.path.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    code, _, _ = run_cmd(f"kaggle kernels output {kernel_slug} -p {log_dir} --logs 2>&1")
    
    if code == 0:
        log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
        if log_files:
            latest_log = max([os.path.join(log_dir, f) for f in log_files], key=os.path.getmtime)
            with open(latest_log, 'r') as f:
                content = f.read()
                if content.strip():
                    print(f"\n📋 Execution Log:")
                    print("-" * 70)
                    # Show last 50 lines if too long
                    lines = content.splitlines()
                    if len(lines) > 50:
                        print(f"... (showing last 50 of {len(lines)} lines)")
                        print("\n".join(lines[-50:]))
                    else:
                        print(content)
                    print("-" * 70)

def main():
    print(f"\n{'='*70}")
    print(f"🚀 KAGGLE AUTOMATION PIPELINE")
    print(f"{'='*70}\n")
    start = time.time()
    
    # Step 1: Push
    print(f"[{timestamp()}] Pushing notebook to Kaggle...")
    os.chdir(WORK_DIR)
    code, out, err = run_cmd("kaggle kernels push -p . 2>&1")
    if code == 0 and "successfully pushed" in out:
        print(f"✅ Pushed successfully\n")
    else:
        print(f"❌ Push failed: {err}\n")
        return
    
    # Step 2: Wait for execution
    print(f"[{timestamp()}] Waiting for notebook execution...")
    start_exec = time.time()
    check = 0
    while time.time() - start_exec < 1200:  # 20 min timeout
        check += 1
        code, out, _ = run_cmd(f"kaggle kernels status {KERNEL_SLUG}")
        elapsed = int(time.time() - start_exec)
        
        if "COMPLETE" in out:
            print(f"✅ Complete after {elapsed}s\n")
            break
        elif "ERROR" in out or "FAILED" in out:
            print(f"❌ Execution failed\n")
            fetch_logs(KERNEL_SLUG, WORK_DIR)
            return
        
        if check % 12 == 1:  # Print every 60 seconds
            print(f"   ⏳ Running... ({elapsed}s)")
        time.sleep(5)
    
    # Fetch logs after completion
    fetch_logs(KERNEL_SLUG, WORK_DIR)
    
    # Step 3: Download
    print(f"[{timestamp()}] Downloading results...")
    code, _, _ = run_cmd(f"kaggle kernels output {KERNEL_SLUG} -p .")
    if code == 0:
        print(f"✅ Downloaded\n")
    else:
        print(f"❌ Download failed\n")
        return
    
    # Step 4: Submit
    print(f"[{timestamp()}] Submitting to competition...")
    submissions_dir = os.path.join(WORK_DIR, 'submissions')
    if os.path.exists(submissions_dir):
        files = sorted([f for f in os.listdir(submissions_dir) if f.endswith('.csv')], reverse=True)
        submission_file = os.path.join(submissions_dir, files[0]) if files else None
    else:
        submission_file = "submission.csv" if os.path.exists("submission.csv") else None
    
    if not submission_file:
        print(f"❌ No submission file found\n")
        return
    
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else f"Auto {datetime.now().strftime('%H:%M')}"
    code, _, _ = run_cmd(f'kaggle competitions submit -c {COMPETITION} -f {submission_file} -m "{msg}" 2>&1')
    print(f"✅ Submitted\n")
    
    # Step 5: Wait for scores
    print(f"[{timestamp()}] Waiting for scores...")
    for i in range(120):
        code, out, _ = run_cmd(f"kaggle competitions submissions -c {COMPETITION} --csv")
        if code == 0 and out:
            lines = out.strip().split('\n')
            if len(lines) >= 2 and "PENDING" not in lines[1]:
                print(f"✅ Scores received!\n")
                print(lines[0])
                print(lines[1])
                if len(lines) > 2:
                    print("\nPrevious:")
                    for prev in lines[2:min(5, len(lines))]:
                        print(prev)
                break
        
        if i % 6 == 0:  # Print every 60 seconds
            print(f"   ⏳ Waiting... ({i*10}s)")
        time.sleep(10)
    
    total = int(time.time() - start)
    print(f"\n{'='*70}")
    print(f"✅ DONE! Total time: {total//60}m {total%60}s")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
