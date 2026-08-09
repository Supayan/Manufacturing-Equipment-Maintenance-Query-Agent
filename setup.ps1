#Requires -Version 5.1
<#
Bootstraps the project: installs uv if missing, creates a .venv on the
pinned Python version, and installs requirements.txt into it.
#>
$ErrorActionPreference = "Stop"

$PythonVersion = "3.11"
$VenvDir = ".venv"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found, installing..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Creating virtual environment (Python $PythonVersion)..."
uv venv --python $PythonVersion $VenvDir

Write-Host "Installing pinned requirements..."
uv pip install -r requirements.txt --python $VenvDir

Write-Host ""
Write-Host "Setup complete. Activate with:"
Write-Host "  $VenvDir\Scripts\activate"
