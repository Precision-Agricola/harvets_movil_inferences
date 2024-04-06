#%%
import torch

print(f"Versión de PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Versión de CUDA: {torch.version.cuda}")