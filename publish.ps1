# BioSuite PyPI Publish Script
# Usage: .\publish.ps1
# Make sure you bumped the version in pyproject.toml first!

Write-Host "=== BioSuite PyPI Publisher ===" -ForegroundColor Green

# Get current version from pyproject.toml
$version = (Get-Content pyproject.toml | Select-String 'version\s*=\s*"(.+)"').Matches.Groups[1].Value
Write-Host "Current version: $version" -ForegroundColor Yellow

# Clean old builds
Write-Host "Cleaning old builds..." -ForegroundColor Cyan
Remove-Item "dist\biosuite-ultra*" -Force -ErrorAction SilentlyContinue
Remove-Item "build" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "biosuite-ultra.egg-info" -Recurse -Force -ErrorAction SilentlyContinue

# Build
Write-Host "Building package..." -ForegroundColor Cyan
python -m build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Upload
Write-Host "Uploading to PyPI..." -ForegroundColor Cyan
twine upload dist/biosuite-ultra-$version*
if ($LASTEXITCODE -ne 0) {
    Write-Host "Upload failed!" -ForegroundColor Red
    exit 1
}

Write-Host "=== Done! ===" -ForegroundColor Green
Write-Host "View at: https://pypi.org/project/biosuite-ultra/$version/" -ForegroundColor Yellow
