param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Repository,
    [Parameter(Mandatory = $true)][string]$Revision,
    [Parameter(Mandatory = $true)][string]$EvidenceDir
)

$ErrorActionPreference = 'Stop'
$root = [System.IO.Path]::GetFullPath($Repository)
$evidence = [System.IO.Path]::GetFullPath($EvidenceDir)
$toolRoot = $PSScriptRoot
$lockPath = 'requirements/compatibility-py312.txt'
$safeRoot = $root.Replace('\', '/')
$commit = & git -c "safe.directory=$safeRoot" -C $root rev-parse "$Revision^{commit}"
if ($LASTEXITCODE -ne 0 -or -not $commit) { throw 'unable to resolve security audit commit' }

New-Item -ItemType Directory -Force -Path $evidence | Out-Null
$blobEvidence = Join-Path $evidence 'lock-blob-identity.json'
$rawAudit = Join-Path $evidence 'pip-audit-raw.json'
$report = Join-Path $evidence 'security-audit.json'

& $Python (Join-Path $toolRoot 'git_blob_sha256.py') `
    --repository $root --revision $commit --path $lockPath `
    --audit-evidence (Join-Path $root 'requirements/security-policy.json') `
    --output $blobEvidence
if ($LASTEXITCODE -ne 0) { throw 'lock Git blob verification failed' }

$versionOutput = & $Python -m pip_audit --version
if ($LASTEXITCODE -ne 0) { throw 'pip-audit is unavailable' }
$auditVersion = ($versionOutput -split '\s+')[-1]
if ($auditVersion -ne '2.10.1') { throw "expected pip-audit 2.10.1, found $auditVersion" }
& $Python -m pip_audit -r (Join-Path $root $lockPath) --format json --output $rawAudit
if ($LASTEXITCODE -notin @(0, 1)) { throw 'pip-audit execution failed' }

& $Python (Join-Path $toolRoot 'create_security_audit.py') `
    --pip-audit-json $rawAudit `
    --policy (Join-Path $root 'requirements/security-policy.json') `
    --blob-identity $blobEvidence `
    --workflow (Join-Path $root '.github/workflows/adaptive-compatibility.yml') `
    --source-commit $commit --audit-tool-version $auditVersion --output $report
if ($LASTEXITCODE -ne 0) { throw 'security audit gate failed' }
