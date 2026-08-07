param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$ArtifactDir,
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [Parameter(Mandatory = $true)][string]$EvidenceDir,
    [Parameter(Mandatory = $true)][string]$SourceCommit
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$artifactPath = [System.IO.Path]::GetFullPath($ArtifactDir)
$workPath = [System.IO.Path]::GetFullPath($WorkDir)
$evidencePath = [System.IO.Path]::GetFullPath($EvidenceDir)
New-Item -ItemType Directory -Force -Path $artifactPath, $workPath, $evidencePath | Out-Null
$versionFile = Join-Path $root 'uqra\_version.py'
$version = & $Python -c "import runpy, sys; print(runpy.run_path(sys.argv[1])['__version__'])" $versionFile
if ($LASTEXITCODE -ne 0 -or -not $version) { throw 'unable to determine UQRA version' }

$safeRoot = $root.Replace('\', '/')
$sourceDateEpoch = & git -c "safe.directory=$safeRoot" -C $root show -s --format=%ct $SourceCommit
if ($LASTEXITCODE -ne 0 -or -not $sourceDateEpoch) {
    throw 'unable to determine source commit timestamp'
}
& (Join-Path $PSScriptRoot 'build_reproducible.ps1') `
    -Python $Python -SourceDir $root -OutputDir $artifactPath `
    -SourceDateEpoch ([long]$sourceDateEpoch) -SourceCommit $SourceCommit

$wheel = Get-ChildItem -LiteralPath $artifactPath -Filter "uqra-$version-*.whl"
$sdist = Get-ChildItem -LiteralPath $artifactPath -Filter "uqra-$version.tar.gz"
if ($wheel.Count -ne 1 -or $sdist.Count -ne 1) { throw 'expected exactly one wheel and one sdist' }

& $Python (Join-Path $PSScriptRoot 'create_distribution_manifest.py') `
    --dist-dir $artifactPath `
    --output (Join-Path $evidencePath 'distribution-manifest.json') `
    --source-commit $SourceCommit `
    --version $version
if ($LASTEXITCODE -ne 0) { throw 'distribution manifest validation failed' }

foreach ($item in @(@{Kind='wheel'; Artifact=$wheel.FullName}, @{Kind='sdist'; Artifact=$sdist.FullName})) {
    $venv = Join-Path $workPath $item.Kind
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "failed to create $($item.Kind) environment" }
    $venvPython = Join-Path $venv 'Scripts\python.exe'
    & $venvPython -m pip install --disable-pip-version-check -r (Join-Path $root 'requirements\compatibility-py312.txt')
    if ($LASTEXITCODE -ne 0) { throw "failed to install locked dependencies for $($item.Kind)" }
    $installArguments = @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-deps')
    if ($item.Kind -eq 'sdist') { $installArguments += '--no-build-isolation' }
    $installArguments += $item.Artifact
    & $venvPython @installArguments
    if ($LASTEXITCODE -ne 0) { throw "failed to install $($item.Kind)" }
    & $venvPython (Join-Path $PSScriptRoot 'verify_installed.py') `
        --source-artifact $item.Artifact `
        --evidence-dir (Join-Path $evidencePath $item.Kind) `
        --install-kind $item.Kind
    if ($LASTEXITCODE -ne 0) { throw "$($item.Kind) clean-install verification failed" }
}
