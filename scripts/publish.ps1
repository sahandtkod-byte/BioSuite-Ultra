# BioSuite PyPI Publish Script
# Usage: .\publish.ps1
# Make sure you bumped the version in pyproject.toml first!

Write-Host "=== BioSuite PyPI Publisher ===" -ForegroundColor Green

# Get current version from pyproject.toml
$version = (Get-Content pyproject.toml | Select-String 'version\s*=\s*"(.+)"').Matches.Groups[1].Value
Write-Host "Current version: $version" -ForegroundColor Yellow

# Clean old builds
Write-Host "Cleaning old builds..." -ForegroundColor Cyan
Remove-Item "dist\*" -Force -ErrorAction SilentlyContinue
Remove-Item "build" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "biosuite*.egg-info" -Recurse -Force -ErrorAction SilentlyContinue

# Build
Write-Host "Building package..." -ForegroundColor Cyan
python -m build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Validate metadata before anything leaves the machine.
Write-Host "Checking package metadata..." -ForegroundColor Cyan
twine check dist/*
if ($LASTEXITCODE -ne 0) {
    Write-Host "twine check failed!" -ForegroundColor Red
    exit 1
}

# Upload. Distribution filenames normalize the project name to underscores
# (PEP 427/625), so the glob must use biosuite_ultra, not biosuite-ultra.
Write-Host "Uploading to PyPI..." -ForegroundColor Cyan
twine upload dist/biosuite_ultra-$version*
if ($LASTEXITCODE -ne 0) {
    Write-Host "Upload failed!" -ForegroundColor Red
    exit 1
}

Write-Host "=== Done! ===" -ForegroundColor Green
Write-Host "View at: https://pypi.org/project/biosuite-ultra/$version/" -ForegroundColor Yellow
