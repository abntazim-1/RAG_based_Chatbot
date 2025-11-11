# Fix for GitHub Large Files Error

## Problem
GitHub rejected your push because model files (417.68 MB) exceed the 100 MB limit.

## Solution: Remove Large Files from Git History

Run these commands in PowerShell (in your project directory):

### Step 1: Remove files from Git tracking (but keep them locally)
```powershell
git rm -r --cached models/
```

### Step 2: Commit the removal
```powershell
git commit -m "Remove large model files from git tracking"
```

### Step 3: Push to GitHub
```powershell
git push origin main
```

## Alternative: If files are already in history

If the files are already in your git history, you need to remove them from history:

### Option A: Use git filter-repo (Recommended)
```powershell
# Install git-filter-repo first (if not installed)
pip install git-filter-repo

# Remove models directory from entire history
git filter-repo --path models/ --invert-paths
```

### Option B: Use BFG Repo-Cleaner
1. Download BFG from: https://rtyley.github.io/bfg-repo-cleaner/
2. Run:
```powershell
java -jar bfg.jar --delete-folders models
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Option C: Reset and recommit (if you haven't pushed yet)
```powershell
# WARNING: This rewrites history - only use if you haven't shared the repo
git reset --soft HEAD~1  # Go back one commit
git reset HEAD models/   # Unstage models
git commit -m "Your commit message without models"
```

## After Fixing

1. The `.gitignore` file has been updated to exclude:
   - `models/` directory
   - `*.safetensors` files
   - Model cache directories
   - Data files

2. Models will be downloaded automatically when needed (they're not needed in the repo)

3. Add a note in README.md that users need to download models:
   ```markdown
   ## Setup Models
   
   Models will be downloaded automatically when you run:
   - `build_embeddings.py` (downloads embedding models)
   - The chunker will use models from `./models/all-mpnet-base-v2`
   ```

## Verify

After fixing, verify with:
```powershell
git status
```

You should see `models/` is now ignored (won't show up in `git status`).

