#!/usr/bin/env python3
"""Commit README, set branch, add remote, and push to GitHub."""
import subprocess
import sys

REPO = "https://github.com/suryansh71234567/reinforcement-learning-on-qwen2.5-1.5b.git"

def run(cmd):
    print(f"> {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.stderr.strip():
        print(r.stderr.strip())
    return r.returncode

# Commit README change
rc = run('git commit -m "first commit"')
if rc != 0:
    print("commit failed (maybe nothing to commit)")

# Rename branch to main
run("git branch -M main")

# Add remote
run(f"git remote add origin {REPO}")

# Push
rc = run("git push -u origin main")
if rc != 0:
    print("push failed")
    sys.exit(1)

print("Done!")
