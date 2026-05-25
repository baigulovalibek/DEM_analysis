# Build the DEM Analyst portable Windows bundle.
#
# Usage:
#   .\packaging\build.ps1            # build only
#   .\packaging\build.ps1 -Zip       # build + zip dist\DEM-Analyst-<ver>.zip
#   .\packaging\build.ps1 -Clean     # nuke build/ and dist/ first

[CmdletBinding()]
param(
    [switch]$Zip,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Venv        = Join-Path $ProjectRoot "env"
$Python      = Join-Path $Venv "Scripts\python.exe"
$PyInstaller = Join-Path $Venv "Scripts\pyinstaller.exe"
$Spec        = Join-Path $PSScriptRoot "DEM-Analyst.spec"
$DistDir     = Join-Path $ProjectRoot "dist\DEM-Analyst"

if (-not (Test-Path $Python)) {
    throw "venv python not found at $Python - create it first (python -m venv env; env\Scripts\pip install -r requirements.txt)"
}

if (-not (Test-Path $PyInstaller)) {
    Write-Host "Installing PyInstaller into venv..." -ForegroundColor Cyan
    & $Python -m pip install --upgrade pyinstaller pyinstaller-hooks-contrib
}

if ($Clean) {
    Write-Host "Cleaning build/ and dist/..." -ForegroundColor Cyan
    Remove-Item -Recurse -Force (Join-Path $ProjectRoot "build")  -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force (Join-Path $ProjectRoot "dist")   -ErrorAction SilentlyContinue
}

Push-Location $ProjectRoot
try {
    Write-Host "Running PyInstaller..." -ForegroundColor Cyan
    # PyInstaller writes INFO to stderr; suppress PowerShell's auto-throw on
    # native stderr so we judge success purely by exit code.
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $logPath = Join-Path $PSScriptRoot "build.log"
    if (Test-Path $logPath) { Remove-Item $logPath }
    & $PyInstaller $Spec --noconfirm --clean 2>&1 | ForEach-Object {
        $line = "$_"
        Add-Content -Path $logPath -Value $line -Encoding utf8
        Write-Host $line
    }
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $prevPref
    if ($rc -ne 0) { throw "PyInstaller failed (exit $rc)" }

    if (-not (Test-Path $DistDir)) { throw "expected $DistDir to exist after build" }

    $bytes = (Get-ChildItem $DistDir -Recurse -File | Measure-Object Length -Sum).Sum
    $mb    = [math]::Round($bytes / 1MB, 1)
    Write-Host ""
    Write-Host "Bundle: $DistDir" -ForegroundColor Green
    Write-Host ("Size  : {0} MB across {1} files" -f $mb, (Get-ChildItem $DistDir -Recurse -File).Count) -ForegroundColor Green

    if ($Zip) {
        $version = (Select-String -Path (Join-Path $ProjectRoot "app\config.py") -Pattern 'APP_VERSION\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
        $zipPath = Join-Path $ProjectRoot "dist\DEM-Analyst-$version-win64.zip"
        if (Test-Path $zipPath) { Remove-Item $zipPath }
        Write-Host "Zipping to $zipPath..." -ForegroundColor Cyan
        Compress-Archive -Path "$DistDir\*" -DestinationPath $zipPath -CompressionLevel Optimal
        $zipMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
        Write-Host ("Zip   : $zipPath ({0} MB)" -f $zipMb) -ForegroundColor Green
    }
}
finally {
    Pop-Location
}
