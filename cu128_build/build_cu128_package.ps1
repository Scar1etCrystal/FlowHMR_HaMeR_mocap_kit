# ============================================================
#  Build the cu128 (RTX 50-series / Blackwell, sm_120) delivery
#  package from a working cu121 pack, without touching that pack.
#
#  Background:
#    sm_120 needs the cu128 wheels. Those same wheels still run on
#    Ampere/Turing/Ada, but cu128 is measurably slower there
#    (~19% on an RTX 3060), so the local pack normally stays on
#    cu121 and cu128 is only used for the delivery build.
#
#  What it does:
#    1. copy the pack to a staging directory
#       (logs / inputs / outputs / video files / __pycache__ excluded)
#    2. install the two cu128 wheels into the staged env
#    3. verify: version must be +cu128, arch list must contain
#       both sm_120 (50-series) and sm_86 (Ampere)
#    4. zip it into 4 GiB volumes
#    5. delete the staging directory
#
#  The working pack is never modified.
#
#  Examples:
#    powershell -ExecutionPolicy Bypass -File build_cu128_package.ps1
#    ... -DryRun      # resolve and print everything, do nothing
#    ... -SkipZip     # prepare + verify the stage, then delete it
#    ... -KeepStage   # keep the prepared stage for inspection
# ============================================================
param(
    [string]$PackRoot = '',
    [string]$StageDir = '',
    [string]$OutDir   = '',
    [string]$ZipName  = 'FlowHMR_kit_50series.zip',
    [string]$SevenZip = '',
    [switch]$SkipZip,
    [switch]$KeepStage,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$kit = $PSScriptRoot

function Say($msg)  { Write-Host $msg }
function Fail($msg) { Write-Host "FAILED: $msg" -ForegroundColor Red; exit 1 }
function HasPack($p) { $p -and (Test-Path -LiteralPath (Join-Path $p 'envs\python.exe')) }

# ---------------------------------------------------------------- locate things
if (-not $PackRoot) {
    foreach ($candidate in @(
        (Split-Path $kit -Parent),
        (Join-Path (Split-Path $kit -Parent) 'flowhmr_pack'),
        'F:\flowhmr_pack'
    )) {
        if (HasPack $candidate) { $PackRoot = $candidate; break }
    }
}
if (-not (HasPack $PackRoot)) {
    Fail 'pack root not found. Pass -PackRoot <dir containing envs\python.exe>'
}

if (-not $StageDir) { $StageDir = (Join-Path (Split-Path $PackRoot -Parent) '_build_cu128') }
if (-not $OutDir)   { $OutDir   = (Join-Path (Split-Path $PackRoot -Parent) 'flowhmr_share') }

if (-not $SevenZip) {
    foreach ($candidate in @(
        'D:\7-Zip\7z.exe',
        'C:\Program Files\7-Zip\7z.exe',
        'C:\Program Files (x86)\7-Zip\7z.exe'
    )) {
        if (Test-Path -LiteralPath $candidate) { $SevenZip = $candidate; break }
    }
    if (-not $SevenZip) {
        $onPath = Get-Command 7z.exe -ErrorAction SilentlyContinue
        if ($onPath) { $SevenZip = $onPath.Source }
    }
}

$torchWheel = Get-ChildItem -LiteralPath $kit -Filter 'torch-*+cu128-*.whl' -ErrorAction SilentlyContinue |
                Select-Object -First 1 -ExpandProperty FullName
$tvWheel    = Get-ChildItem -LiteralPath $kit -Filter 'torchvision-*+cu128-*.whl' -ErrorAction SilentlyContinue |
                Select-Object -First 1 -ExpandProperty FullName
$tvWheel = @($tvWheel)[0]

Say "pack root   : $PackRoot"
Say "staging     : $StageDir"
Say "output      : $(Join-Path $OutDir $ZipName)"
Say "7-Zip       : $(if ($SevenZip) { $SevenZip } else { '(not found yet)' })"
Say "torch       : $(if ($torchWheel) { Split-Path $torchWheel -Leaf } else { '(missing)' })"
Say "torchvision : $(if ($tvWheel) { Split-Path $tvWheel -Leaf } else { '(missing)' })"

if ($DryRun) { Say ''; Say 'dry run: nothing was created.'; exit 0 }

foreach ($need in @(
    @{p = $torchWheel; what = 'torch cu128 wheel (see REBUILD.md for the download command)'},
    @{p = $tvWheel;    what = 'torchvision cu128 wheel'},
    @{p = $SevenZip;   what = '7-Zip (pass -SevenZip <7z.exe>)'}
)) {
    if (-not $need.p -or -not (Test-Path -LiteralPath $need.p)) { Fail "missing $($need.what)" }
}

$free = (Get-PSDrive (Split-Path $StageDir -Qualifier).TrimEnd(':')).Free
Say ("free space on the staging drive: {0:N1} GB (about 70 GB needed)" -f ($free / 1GB))
if ($free -lt 70GB) { Fail 'not enough free space for staging + output' }

# ---------------------------------------------------------------- 1. stage
if (Test-Path -LiteralPath $StageDir) {
    Say 'removing an older stage...'
    Remove-Item -LiteralPath $StageDir -Recurse -Force
}
Say 'copying the pack to staging (logs / inputs / outputs / videos / __pycache__ excluded)...'
$robolog = & robocopy $PackRoot $StageDir /E /NFL /NDL /NJH /NJS /NP `
    /XD "$PackRoot\logs" "$PackRoot\inputs" "$PackRoot\outputs" '__pycache__' `
    /XF '*.mp4' '*.mkv' '*.mov' '*.avi' '*.webm' '*.llc' 2>&1
if ($LASTEXITCODE -ge 8) { $robolog | Select-Object -Last 15; Fail "robocopy failed ($LASTEXITCODE)" }
Say ("  staging size: {0:N1} GB" -f ((Get-ChildItem $StageDir -Recurse -File -Force |
        Measure-Object Length -Sum).Sum / 1GB))

# ---------------------------------------------------------------- 2. cu128
Say 'installing the cu128 wheels into the staging env...'
& "$StageDir\envs\python.exe" -m pip install --no-deps --force-reinstall $torchWheel $tvWheel |
    Select-Object -Last 2

# ---------------------------------------------------------------- 3. verify
Say 'verifying the staged environment...'
$check = & "$StageDir\envs\python.exe" -c @"
import torch, torchvision
arch = torch.cuda.get_arch_list()
print('torch', torch.__version__, 'torchvision', torchvision.__version__)
print('cuda', torch.version.cuda)
print('arch', arch)
assert torch.__version__.endswith('+cu128'), 'torch is not the cu128 build'
assert 'sm_120' in arch, 'sm_120 missing - would fail on RTX 50-series'
assert 'sm_86' in arch, 'sm_86 missing - would fail on Ampere'
print('OK')
"@ 2>&1
$check | ForEach-Object { "  $_" }
if ($LASTEXITCODE -ne 0) { Fail 'the staged torch is not usable' }

# ---------------------------------------------------------------- 4. zip
if (-not $SkipZip) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Get-ChildItem -LiteralPath $OutDir -Filter "$ZipName*" -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item $_.FullName -Force }
    Say "archiving to $(Join-Path $OutDir $ZipName) (4 GiB volumes)..."
    Push-Location $StageDir
    try {
        & $SevenZip a -tzip -mx1 -mmt=on -v4g -bso0 -bsp0 (Join-Path $OutDir $ZipName) '*' '-xr!__pycache__'
        if ($LASTEXITCODE -ne 0) { Fail "7-Zip failed ($LASTEXITCODE)" }
    } finally { Pop-Location }
    Say ''
    Say 'built:'
    Get-ChildItem -LiteralPath $OutDir -Filter "$ZipName*" |
        ForEach-Object { "  {0}  {1:N2} GB" -f $_.Name, ($_.Length / 1GB) }
    Say ("total {0:N2} GB" -f ((Get-ChildItem $OutDir -Filter "$ZipName*" |
            Measure-Object Length -Sum).Sum / 1GB))
}

# ---------------------------------------------------------------- 5. clean
if ($KeepStage) {
    Say "stage kept at $StageDir"
} else {
    Say 'removing the stage...'
    Remove-Item -LiteralPath $StageDir -Recurse -Force
}
Say 'done.'
