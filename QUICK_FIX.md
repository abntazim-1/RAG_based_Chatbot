# Quick Fix for GitHub Large Files Error

## The Problem
The large model files (417.68 MB) are still in your git history, even though you removed them from the current commit. GitHub checks the entire history.

## Solution: Remove from Git History

### Option 1: Use the PowerShell Script (Easiest)

1. Run the provided script:
```powershell
.\remove_large_files.ps1
```

2. Then force push:
```powershell
git push origin main --force
```

⚠️ **Warning**: `--force` overwrites remote history. Only use if you're the only one working on this repo.

---

### Option 2: Manual Commands

Run these commands one by one in PowerShell:

```powershell
# 1. Remove from current tracking
git rm -r --cached models/

# 2. Remove from entire git history
git filter-branch --force --index-filter "git rm -rf --cached --ignore-unmatch models/" --prune-empty --tag-name-filter cat -- --all

# 3. Clean up
git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 4. Force push (WARNING: overwrites remote)
git push origin main --force
```

---

### Option 3: Use git-filter-repo (Recommended for future)

If you have Python installed:

```powershell
# Install git-filter-repo
pip install git-filter-repo

# Remove models from history
git filter-repo --path models/ --invert-paths

# Force push
git push origin main --force
```

---

## After Fixing

1. ✅ Verify models are ignored:
```powershell
git status
```
You should NOT see `models/` in the output.

2. ✅ Verify files are still on your computer:
```powershell
Test-Path models/
```
Should return `True` (files are still there locally, just not in git).

3. ✅ Update README.md to mention that models need to be downloaded:
```markdown
## Setup

Models will be automatically downloaded when you run:
- `build_embeddings.py` - downloads embedding models
- The chunker uses models from `./models/all-mpnet-base-v2`
```

---

## Why This Happened

- You committed large model files to git
- Even after removing them, they remain in git history
- GitHub checks the entire history, not just current files
- Solution: Remove from history completely

---

## Prevention

Your `.gitignore` already has `models/` in it, so future commits won't include model files. The issue was files already in history.

