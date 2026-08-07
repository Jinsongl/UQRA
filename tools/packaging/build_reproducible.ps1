param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$SourceDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [Parameter(Mandatory = $true)][long]$SourceDateEpoch,
    [string]$SourceCommit
)

$ErrorActionPreference = 'Stop'
$sourcePath = [System.IO.Path]::GetFullPath($SourceDir)
$outputPath = [System.IO.Path]::GetFullPath($OutputDir)
$stagedSource = Join-Path $outputPath '.source'
$stagedOutput = Join-Path $outputPath '.artifacts'

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
    $prepareArguments = @(
        (Join-Path $PSScriptRoot 'prepare_build_source.py'),
        '--source', $sourcePath,
        '--destination', $stagedSource
    )
    if ($SourceCommit) { $prepareArguments += @('--revision', $SourceCommit) }
    & $Python @prepareArguments
    if ($LASTEXITCODE -ne 0) { throw 'isolated source preparation failed' }

    $env:SOURCE_DATE_EPOCH = $SourceDateEpoch.ToString([Globalization.CultureInfo]::InvariantCulture)
    $env:PYTHONHASHSEED = '0'
    & $Python -m build --no-isolation --outdir $stagedOutput $stagedSource
    if ($LASTEXITCODE -ne 0) { throw 'distribution build failed' }
    $sdist = @(Get-ChildItem -LiteralPath $stagedOutput -Filter '*.tar.gz')
    if ($sdist.Count -ne 1) { throw 'expected exactly one sdist before normalization' }
    & $Python (Join-Path $PSScriptRoot 'normalize_sdist.py') $sdist[0].FullName --epoch $SourceDateEpoch
    if ($LASTEXITCODE -ne 0) { throw 'sdist normalization failed' }
    Get-ChildItem -LiteralPath $stagedOutput -File | Move-Item -Destination $outputPath
}
finally {
    $env:SOURCE_DATE_EPOCH = $previousEpoch
    $env:PYTHONHASHSEED = $previousHashSeed
    foreach ($temporary in @($stagedSource, $stagedOutput)) {
        $resolvedOutput = [System.IO.Path]::GetFullPath($outputPath).TrimEnd('\') + '\'
        $resolvedTemporary = [System.IO.Path]::GetFullPath($temporary)
        if (-not $resolvedTemporary.StartsWith($resolvedOutput, [StringComparison]::OrdinalIgnoreCase)) {
            throw "refusing to clean path outside build output: $resolvedTemporary"
        }
        if (Test-Path -LiteralPath $resolvedTemporary) {
            Remove-Item -LiteralPath $resolvedTemporary -Recurse -Force
        }
    }
}

$artifacts = @(Get-ChildItem -LiteralPath $outputPath -File)
if (($artifacts | Where-Object Extension -eq '.whl').Count -ne 1 -or
    ($artifacts | Where-Object Name -Like '*.tar.gz').Count -ne 1 -or
    $artifacts.Count -ne 2) {
    throw 'expected exactly one wheel and one sdist'
}
