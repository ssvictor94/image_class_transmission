# 下载 DeiT 预训练权重（绕过 Hugging Face，使用 Facebook CDN）
$destDir = Join-Path $PSScriptRoot "pretrained_models"
$dest = Join-Path $destDir "deit_tiny_patch16_224-a1311bcf.pth"
New-Item -ItemType Directory -Force -Path $destDir | Out-Null

if (Test-Path $dest) {
    Write-Host "权重已存在: $dest"
    exit 0
}

$url = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
Write-Host "下载 $url ..."
Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
Write-Host "完成: $dest ($((Get-Item $dest).Length) bytes)"
