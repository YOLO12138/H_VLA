# HVLA

> Forked from [starVLA](https://github.com/starVLA/starVLA).

## Quickstart

This guide sets up **HVLA + ManiFlow** in a single conda environment.

## Prerequisites

- Linux with NVIDIA GPU
- CUDA toolkit 12.x installed (`nvcc -V` to verify)
- Conda (Miniconda or Anaconda)
- Vulkan (for ManiFlow simulation envs):
  ```bash
  sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
  ```

---

## 1. Create Environment & Install Core Dependencies

```bash
# Create conda env
conda create -n hvla python=3.11 -y
conda activate hvla

# Install all dependencies (HVLA + ManiFlow)
pip install -r requirements.txt
```

## 2. Install FlashAttention & PyTorch3D

```bash
# FlashAttention2
pip install flash-attn --no-build-isolation
```

### PyTorch3D

PyTorch3D must be compiled from source against your current PyTorch CUDA version. If your system CUDA toolkit version (check with `nvcc -V` or `ls /usr/local/ | grep cuda`) differs from the one PyTorch was built with (check with `python -c "import torch; print(torch.version.cuda)"`), you need to install a matching toolkit first:

```bash
# Example: PyTorch was compiled with CUDA 12.8 but system has CUDA 13.2
# Install matching CUDA toolkit via conda
conda install -c nvidia cuda-toolkit=12.8 -y

# Build pytorch3d using the conda CUDA toolkit
CUDA_HOME=$CONDA_PREFIX pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
```

If your system CUDA version already matches PyTorch's, you can simply run:
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## 3. Install Packages in Editable Mode

```bash
# Install HVLA
pip install -e .

# Install ManiFlow
pip install -e hvla/model/modules/maniflow/ManiFlow
```