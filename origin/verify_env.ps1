# 环境自检脚本
$Python = "C:\Users\hp\.conda\envs\djscc_image_tranmission\python.exe"
Set-Location $PSScriptRoot

Write-Host "=== Python / PyTorch ==="
& $Python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host "`n=== 依赖 import ==="
& $Python -c @"
import hydra, timm, omegaconf, ptflops, matplotlib
from torchvision.datasets import Imagenette
print('deps OK')
"@

Write-Host "`n=== 数据集 ==="
$data = ".\data\imagenette\imagenette2"
if (Test-Path $data) {
    Write-Host "data path OK: $data"
    & $Python -c @"
from torchvision.datasets import Imagenette
from torchvision import transforms
t = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
ds = Imagenette(root='./data/imagenette', split='train', size='full', download=False, transform=t)
print('train samples', len(ds))
"@
} else {
    Write-Host "WARN: 数据集链接不存在，请运行 setup_data.ps1"
}

Write-Host "`n=== 官方代码 import ==="
& $Python -c @"
from methods.proposal import SemanticVit, AdaptiveBlock
from comm.channel import GaussianNoiseChannel
from utils import CommunicationPipeline
print('official modules OK')
"@
