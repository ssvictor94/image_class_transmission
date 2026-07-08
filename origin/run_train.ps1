# 论文 Proposal + JSCC 训练（Imagenette + ViT）
# 用法: .\run_train.ps1
# 参考: bash/slurm/imagenette_jscc_margin_final.sh

$Python = "C:\Users\hp\.conda\envs\djscc_image_tranmission\python.exe"
Set-Location $PSScriptRoot

$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_OFFLINE = "1"

# 确保本地 DeiT 权重存在（不依赖 Hugging Face）
$weight = Join-Path $PSScriptRoot "pretrained_models\deit_tiny_patch16_224-a1311bcf.pth"
if (-not (Test-Path $weight)) {
    Write-Host ">>> 下载预训练权重..."
    & (Join-Path $PSScriptRoot "download_pretrained.ps1")
}

& $Python main.py `
  training_pipeline=imagenette224_vit16 `
  pretraining_pipeline=imagenette224 `
  model=deit_tiny_patch16_224 `
  +jscc=proposal `
  method=proposal `
  method.loss.inner_flops_type=margin `
  method.loss.inner_flops_w=1 `
  method.loss.output_flops_w=2 `
  final_evaluation=semantic `
  +method.model.blocks_to_transform=6 `
  comm_evaluation=semantic `
  "serialization.values_to_prepend=[jscc]" `
  device=0
