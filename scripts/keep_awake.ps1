# Keeps Windows from sleeping while long training runs are going (unattended nights).
# Uses SetThreadExecutionState (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); no power settings are changed, and the request
# ends automatically when this process exits.
#
#   Start-Process powershell -ArgumentList "-NoProfile","-File","scripts\keep_awake.ps1","-Hours","10" -WindowStyle Hidden
#
# Stops after -Hours, or earlier when the file results\.stop_keep_awake exists.
param([double]$Hours = 10)

$signature = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$power = Add-Type -MemberDefinition $signature -Name Power -Namespace Win32 -PassThru
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"

$power::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
$end = (Get-Date).AddHours($Hours)
$stopFile = Join-Path $PSScriptRoot "..\results\.stop_keep_awake"
while ((Get-Date) -lt $end -and -not (Test-Path $stopFile)) {
    Start-Sleep -Seconds 60
}
$power::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
