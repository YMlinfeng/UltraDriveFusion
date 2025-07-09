import torch
from torch import nn
import numpy as np

from diffusers import AutoencoderKLCogVideoX

# vae = AutoencoderKLCogVideoX.from_pretrained("/mnt/bn/occupancy3d/workspace/mzj/CogVideoX-2b", subfolder="vae", torch_dtype=torch.float16).to("cuda")
vae = AutoencoderKLCogVideoX().to("cuda")
x = torch.randn(1, 3, 25, 64, 64).to("cuda")  # shape: (B, C, T, H, W)
with torch.no_grad():
    posterior = vae.encode(x).latent_dist
    z = posterior.sample()
    recon_video = vae.decode(z).sample

# print("posterior shape:", posterior.sample().shape)
print("Latent vector shape:", z.shape)
print("Reconstructed video shape:", recon_video.shape)