# PowerShell script to remove large model files from git history
# Run this script in your project directory

Write-Host "Removing large model files from git history..." -ForegroundColor Yellow

# Step 1: Remove models directory from git tracking (keep files locally)
Write-Host "`nStep 1: Removing models/ from git tracking..." -ForegroundColor Cyan
git rm -r --cached models/

# Step 2: Use git filter-branch to remove from entire history
Write-Host "`nStep 2: Removing from git history (this may take a while)..." -ForegroundColor Cyan
Write-Host "WARNING: This rewrites git history!" -ForegroundColor Red

# Remove models directory from all commits
git filter-branch --force --index-filter "git rm -rf --cached --ignore-unmatch models/" --prune-empty --tag-name-filter cat -- --all

# Step 3: Clean up
Write-Host "`nStep 3: Cleaning up..." -ForegroundColor Cyan
git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

Write-Host "`n✅ Done! Large files removed from git history." -ForegroundColor Green
Write-Host "`nNow you can push to GitHub:" -ForegroundColor Yellow
Write-Host "  git push origin main --force" -ForegroundColor White
Write-Host "`n⚠️  WARNING: Using --force will overwrite remote history!" -ForegroundColor Red
Write-Host "Only do this if you're sure no one else has cloned the repo." -ForegroundColor Yellow

