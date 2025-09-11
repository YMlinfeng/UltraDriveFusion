## 1 Introduction

...

## 2 Related Works

### 2.1 Generative World Models

Diffusion-based world models fall into two families:

- **UNet diffusers** process each frame using 2D UNets (Ronneberger, Fischer, and Brox, 2015) and couple time via 3D convolutions or temporal attention. Representative models include:
  - MyGo (Yao et al., 2024)
  - MagicDrive (Gao et al., 2023)
  - DriveScape (Wu et al., 2024)
  - Delphi (Ma et al., 2024)

- **DiT diffusers** use Transformers to jointly learn spatial–temporal–view correlations. Notable examples are:
  - MagicDriveDiT (Gao et al., 2024a)
  - DiVE (Jiang et al., 2024)
  - Panacea (Wen et al., 2024)

UNet models are lightweight but suffer from **receptive-field drift**, while DiT models preserve **long-range structure** at the cost of increased memory, often relying on **progressive curricula**.

Orthogonally, **autoregressive simulators** decode frames sequentially based on action sequences, providing explicit trajectory control but introducing **Markov error**, which limits high-resolution, multi-view synthesis. Examples include:

- DriveGAN (Kim et al., 2021)
- DriveDreamer (Wang et al., 2024a)
- UMGen (Wu et al., 2025)
- DreamForge (Mei et al., 2024)

---

### 2.2 Controllable Multi-View Video Generation

Diffusion-based video control has evolved from **single-branch ControlNet designs** to **unified multimodal frameworks**, integrating structure, identity, image, time, audio, and text using a shared backbone.

Examples include:

- **CineMaster** (Wang et al., 2025): Enables 3D GUI edits
- **SketchVideo** (Liu et al., 2025): Constrains motion via line drawings
- Large frameworks like:
  - **Wan 2.1** (Wan et al., 2025)
  - **HunyuanVideo** (Kong et al., 2024)
  - **TokenFlow** (Geyer et al., 2023): Fuse text, depth, and optical flow to enable "train once, reuse everywhere"

In autonomous driving, models must additionally respect **pixel-aligned geometry** (e.g., HD maps, trajectories, camera poses). Early efforts like:

- **MagicMotion** (Li et al., 2025)
- **MotionCtrl** (Wang et al., 2024c)

adapt generic image/text-to-video (I2V/T2V) pipelines but only support **one–two views**, leaving **six-camera setups** unaddressed.

Current strategies for **cross-view coherence** include:

1. **Latent-Space Sharing**  
   - GAIA (Russell et al., 2025)  
   - DreamFusion (Poole et al., 2022)  
   *Drawback:* prone to drift across views.

2. **Geometry-Aware Modelling**  
   - MyGo (Yao et al., 2024)  
   - DriveDreamer-2 (Zhao et al., 2025b)  
   - NVS-Diffusion (You et al., 2024)  
   *Approach:* use extrinsics, epipolar masks, or NeRF/voxel fields  
   *Trade-off:* requires calibration and significant compute.

3. **View-Aware Attention**  
   - VideoComposer (Wang et al., 2023)  
   - DiVE (Jiang et al., 2024)  
   - UniMLVG (Chen et al., 2024)  
   *Benefit:* calibration-free  
   *Limitation:* relies on late-fusion latents and is memory-intensive.

Further refinements include:

- **MVDiffusion** (Deng et al., 2023)
- **Vivid-Zoo** (Li et al., 2024a)  
*Enhancements:* Add epipolar or 2D/3D alignment during decoding, though **texture and lighting mismatches** remain.

---

### Motivation for Our Approach

These limitations motivate our **unified compression design** that:

- **Concatenates six camera streams before encoding**, enabling a single 3D convolutional grid to capture full scene geometry.
- **Incorporates unified control tokens**, significantly improving:
  - **Subject consistency (SC)**
  - **Photometric consistency (PC)**  
while preserving **fine-grained controllability**.

--- 

