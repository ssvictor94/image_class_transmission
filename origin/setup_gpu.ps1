# RTX 5090 (sm_120) 需要 PyTorch cu128，官方 requirements 里的 cu124 不支持 Blackwell
# 用法: 在 PowerShell 中运行 .\setup_gpu.ps1

$Python = "C:\Users\hp\.conda\envs\djscc_image_tranmission\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "找不到 conda 环境 djscc_image_tranmission"
    exit 1
}

Write-Host ">>> 安装/升级 PyTorch cu128 (RTX 5090)..."
& $Python -m pip install --upgrade pip
& $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --upgrade

Write-Host ">>> 安装官方其余依赖..."
& $Python -m pip install hydra-core==1.3.2 huggingface-hub==0.27.1 tqdm==4.67.1 omegaconf==2.3.0 timm matplotlib ptflops pillow

Write-Host ">>> 验证 GPU..."
& $Python -c @"
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
    x = torch.tensor([2.0], device='cuda')
    print('smoke test', (x * x).item())
"@

Write-Host "完成。若 smoke test 输出 4.0 则 GPU 可用。"
