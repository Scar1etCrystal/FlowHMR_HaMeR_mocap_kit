param(
  [Parameter(Mandatory=$true,Position=0)][string]$Video,
  [Parameter(Mandatory=$true,Position=1)][string]$MainTrack,
  [Parameter(Mandatory=$true,Position=2)][string]$OutDir,
  [Parameter(ValueFromRemainingArguments=$true)][string[]]$Extra
)
$pack = Split-Path $PSScriptRoot -Parent
& "$pack\envs\python.exe" "$PSScriptRoot\hamer_pose_sampled.py" --video $Video --main_track $MainTrack --out_dir $OutDir @Extra
exit $LASTEXITCODE
