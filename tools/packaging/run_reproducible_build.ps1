param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$SourceDir,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [Parameter(Mandatory = $true)][long]$SourceDateEpoch,
    [Parameter(Mandatory = $true)][string]$SourceCommit,
    [string]$EvidencePath
)

$ErrorActionPreference = 'Stop'
$toolDir = $PSScriptRoot
$outputPath = [System.IO.Path]::GetFullPath($OutputRoot)
$first = Join-Path $outputPath 'first'
$second = Join-Path $outputPath 'second'
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

& (Join-Path $toolDir 'build_reproducible.ps1') -Python $Python -SourceDir $SourceDir `
    -OutputDir $first -SourceDateEpoch $SourceDateEpoch -SourceCommit $SourceCommit
& (Join-Path $toolDir 'build_reproducible.ps1') -Python $Python -SourceDir $SourceDir `
    -OutputDir $second -SourceDateEpoch $SourceDateEpoch -SourceCommit $SourceCommit

$arguments = @(
    (Join-Path $toolDir 'verify_reproducible_build.py'),
    '--first', $first,
    '--second', $second,
    '--source-commit', $SourceCommit,
    '--source-date-epoch', $SourceDateEpoch
)
if ($EvidencePath) { $arguments += @('--output', $EvidencePath) }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'byte-level reproducibility verification failed' }
