# Spill Sense - Frontend Runner
Set-Location -Path "$PSScriptRoot\frontend"
$env:TEMP = "D:\tmp"
$env:TMP = "D:\tmp"
$env:npm_config_cache = "D:\tmp\npm_cache"

Write-Host "Node: $(node --version)"
Write-Host "NPM:  $(npm.cmd --version)"

Write-Host "Starting Vite dev server on port 5173..."
npm.cmd run dev -- --host

