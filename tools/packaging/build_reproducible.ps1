param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$SourceDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [Parameter(Mandatory = $true)][long]$SourceDateEpoch
)

$ErrorActionPreference = 'Stop'
$sourcePath = [System.IO.Path]::GetFullPath($SourceDir)
$outputPath = [System.IO.Path]::GetFullPath($OutputDir)

if ($SourceDateEpoch -lt 315532800) {
    throw 'SourceDateEpoch must be at or after 1980-01-01 for ZIP compatibility'
}

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
if ((Get-ChildItem -LiteralPath $outputPath -Force).Count -ne 0) {
    throw "reproducible build output directory must be empty: $outputPath"
}

$previousEpoch = $env:SOURCE_DATE_EPOCH
$previousHashSeed = $env:PYTHONHASHSEED
try {
    $env:SOURCE_DATE_EPOCH = $SourceDateEpoch.ToString([Globalization.CultureInfo]::InvariantCulture)
    $env:PYTHONHASHSEED = '0'
    & $Python -m build --no-isolation --outdir $outputPath $sourcePath
    if ($LASTEXITCODE -ne 0) { throw 'distribution build failed' }
    $sdist = @(Get-ChildItem -LiteralPath $outputPath -Filter '*.tar.gz')
    if ($sdist.Count -ne 1) { throw 'expected exactly one sdist before normalization' }
    & $Python (Join-Path $PSScriptRoot 'normalize_sdist.py') $sdist[0].FullName --epoch $SourceDateEpoch
    if ($LASTEXITCODE -ne 0) { throw 'sdist normalization failed' }
}
finally {
    $env:SOURCE_DATE_EPOCH = $previousEpoch
    $env:PYTHONHASHSEED = $previousHashSeed
}

$artifacts = @(Get-ChildItem -LiteralPath $outputPath -File)
if (($artifacts | Where-Object Extension -eq '.whl').Count -ne 1 -or
    ($artifacts | Where-Object Name -Like '*.tar.gz').Count -ne 1 -or
    $artifacts.Count -ne 2) {
    throw 'expected exactly one wheel and one sdist'
}
