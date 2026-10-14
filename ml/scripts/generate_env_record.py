import sys
import platform
import subprocess
import torch
import onnxruntime
import numpy

lines = [
    "=" * 80,
    "SPILLSENSE (SIH-26143) REPRODUCTION ENVIRONMENT RECORD",
    "=" * 80,
    f"OS: {platform.platform()}",
    f"Architecture: {platform.machine()}",
    f"Processor: {platform.processor()}",
    f"Python Version: {sys.version}",
    f"PyTorch Version: {torch.__version__}",
    f"CUDA Available: {torch.cuda.is_available()}",
    f"CUDA Version: {torch.version.cuda}",
    f"cuDNN Version: {torch.backends.cudnn.version()}",
    f"NVIDIA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}",
    f"ONNX Runtime Version: {onnxruntime.__version__}",
    f"NumPy Version: {numpy.__version__}",
    "=" * 80,
    "PIP FREEZE:",
    "=" * 80,
]

freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8")
lines.append(freeze)

with open("ml/reports/final_model_audit/reproduction_environment.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("[SUCCESS] reproduction_environment.txt written successfully")
