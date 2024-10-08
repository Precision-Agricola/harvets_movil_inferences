#%%
import torch
from wasabi import msg


msg.info(f"Versión de PyTorch: {torch.__version__}")
msg.info(f"Versión de CUDA: {torch.version.cuda}")

if torch.cuda.is_available():
    msg.good("CUDA is available")
elif torch.backends.mps.is_available():
    msg.good("MPS is available")
else:
    msg.warn("CPU only")

