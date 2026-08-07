param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$SourceDir,
    [Parameter(Mandatory = $true)][string]$ScratchRoot,
    [Parameter(Mandatory = $true)][long]$SourceDateEpoch,
    [string]$EvidencePath
)

$ErrorActionPreference = 'Stop'
$sourcePath = [System.IO.Path]::GetFullPath($SourceDir)
$scratchPath = [System.IO.Path]::GetFullPath($ScratchRoot)
$toolRelative = 'tools\packaging'
$cases = @(
    @{ Name = 'space'; Directory = 'repository with space' },
    @{ Name = 'single_quote'; Directory = "repository's quote" },
    @{ Name = 'non_ascii'; Directory = '仓库非ASCII' }
)
$safeDirectory = $sourcePath.Replace('\', '/')
$sourceFiles = @(& git -c "safe.directory=$safeDirectory" -C $sourcePath `
    ls-files --cached --others --exclude-standard)
if ($LASTEXITCODE -ne 0 -or $sourceFiles.Count -eq 0) {
    throw 'unable to enumerate source files for path regression'
}

New-Item -ItemType Directory -Force -Path $scratchPath | Out-Null
$reports = @()
foreach ($case in $cases) {
    $caseRoot = Join-Path $scratchPath $case.Directory
    $repository = Join-Path $caseRoot 'source'
    $distribution = Join-Path $caseRoot 'distribution output'
    $environment = Join-Path $caseRoot 'clean install environment'
    $evidence = Join-Path $caseRoot 'smoke evidence'
    if (Test-Path -LiteralPath $caseRoot) {
        throw "path regression case directory must not already exist: $caseRoot"
    }
    New-Item -ItemType Directory -Force -Path $repository | Out-Null
    foreach ($relative in $sourceFiles) {
        $sourceFile = Join-Path $sourcePath $relative
        $targetFile = Join-Path $repository $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $targetFile) | Out-Null
        Copy-Item -LiteralPath $sourceFile -Destination $targetFile -Force
    }

    $versionFile = Join-Path $repository 'uqra\_version.py'
    $version = & $Python -c "import runpy, sys; print(runpy.run_path(sys.argv[1])['__version__'])" $versionFile
    if ($LASTEXITCODE -ne 0 -or -not $version) { throw "version discovery failed for $($case.Name)" }

    & $Python (Join-Path $repository "$toolRelative\normalize_sdist.py") --help | Out-Null
    & (Join-Path $repository "$toolRelative\build_reproducible.ps1") `
        -Python $Python -SourceDir $repository -OutputDir $distribution `
        -SourceDateEpoch $SourceDateEpoch
    $wheel = @(Get-ChildItem -LiteralPath $distribution -Filter "uqra-$version-*.whl")
    if ($wheel.Count -ne 1) { throw "wheel discovery failed for $($case.Name)" }

    & $Python -m venv $environment
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed for $($case.Name)" }
    $venvPython = Join-Path $environment 'Scripts\python.exe'
    & $venvPython -m pip install --disable-pip-version-check -r `
        (Join-Path $repository 'requirements\compatibility-py312.txt')
    if ($LASTEXITCODE -ne 0) { throw "locked dependency install failed for $($case.Name)" }
    & $venvPython -m pip install --disable-pip-version-check --no-deps $wheel[0].FullName
    if ($LASTEXITCODE -ne 0) { throw "wheel install failed for $($case.Name)" }
    & $venvPython (Join-Path $repository "$toolRelative\verify_installed.py") `
        --source-artifact $wheel[0].FullName --evidence-dir $evidence --install-kind wheel
    if ($LASTEXITCODE -ne 0) { throw "installed smoke failed for $($case.Name)" }

    $reports += @{
        case = $case.Name
        repository_path = $repository
        version = $version
        wheel = $wheel[0].Name
        status = 'passed'
    }
}

$report = @{
    schema = 'uqra.packaging.windows-path-regression/v1'
    cases = $reports
}
$json = $report | ConvertTo-Json -Depth 5
if ($EvidencePath) {
    $target = [System.IO.Path]::GetFullPath($EvidencePath)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    [System.IO.File]::WriteAllText($target, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
}
$json
