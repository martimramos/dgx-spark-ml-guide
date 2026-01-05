# DGX Spark PyTorch & CUDA Guide: Running ML on GB10 Blackwell

> **The definitive guide to running PyTorch and ML workloads on NVIDIA DGX Spark with GB10 GPU.**
> Written by someone who spent 72+ hours debugging so you don't have to.

---

## Document Information

| | |
|---------------------|----------------------------------------------------------------------|
| **Author** | **Martim Ramos** , DevOps @ AXA |
| **Hardware** | NVIDIA DGX Spark with GB10 (Blackwell architecture, sm_121) |
| **Last Updated** | January 2025 |
| **Verified Working** | Video generation, lip-sync avatars, image diffusion, music generation |
| **License** | MIT |

---

## Why This Guide Exists

**I bought a DGX Spark expecting to run video generation models within hours. Instead, I spent three days fighting errors that had zero documentation online.**

The DGX Spark is NVIDIA's first desktop AI supercomputer featuring the GB10 Blackwell chip. It's powerful 128GB unified memory, 1 PFLOP of AI performance but almost nothing works out of the box:

- ❌ Standard PyTorch doesn't support it
- ❌ NGC containers don't support it  
- ❌ Flash Attention has no prebuilt wheels
- ❌ Most ARM64 + CUDA packages don't exist

This guide documents every problem I encountered and how I solved it. If you're Googling a DGX Spark error at 2am, this is for you.

---

## Quick Start: What You Need to Know in 60 Seconds

**The core issue:** GB10 uses Blackwell architecture (sm_121), which requires PyTorch nightly builds. The sm_120 target in PyTorch nightly is binary compatible with sm_121.

**The solution for 90% of projects:**

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**If you're using MMLab (mmcv, mmdet, mmpose):** You need Docker with Python 3.10. See [Challenge 4](#challenge-4-python-312-breaks-mmlab-stack).

**If builds fail with CUDA errors:** Your system has CUDA 13.0 but PyTorch uses 12.8. See [Challenge 2](#challenge-2-cuda-version-mismatch-when-compiling-extensions).

---

## System Specifications

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GB10 (Blackwell architecture, compute capability sm_121) |
| **Platform** | ARM64 (aarch64) not x86_64 |
| **System CUDA** | 13.0 |
| **Unified Memory** | 128GB shared between CPU and GPU |
| **OS** | Ubuntu 24.04 LTS |
| **Driver** | 580.95.05 |

**Key insight:** The 128GB is unified memory CPU and GPU share the same pool. This means CPU offloading provides zero benefit (unlike discrete GPUs), but you can load massive models without OOM errors.

---

## Challenge 1: PyTorch Doesn't Detect the GPU

### What error will I see?

```python
>>> import torch
>>> torch.cuda.is_available()
False
```

Or errors mentioning "unsupported compute capability" or "sm_121".

### Why does this happen?

Standard PyTorch releases (2.5.x and earlier) only support up to sm_90 (Hopper architecture). The GB10 uses sm_121 (Blackwell), which is only supported in PyTorch nightly builds with CUDA 12.8.

### How do I fix it?

Install PyTorch nightly with CUDA 12.8:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

The sm_120 architecture in these builds is binary compatible with sm_121.

### How do I verify it's working?

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

**Expected output:**

```
PyTorch: 2.11.0.dev2025XXXX+cu128
CUDA: True
Device: NVIDIA GB10
```

---

## Challenge 2: CUDA Version Mismatch When Compiling Extensions

### What error will I see?

```
ModuleNotFoundError: No module named 'mmcv._ext'
```

Or during compilation:

```
RuntimeError: The detected CUDA version (13.0) mismatches the version that was used to compile PyTorch (12.8)
```

### Why does this happen?

Your DGX Spark has CUDA 13.0 installed system-wide, but PyTorch nightly bundles CUDA 12.8. When compiling CUDA extensions (mmcv, flash_attn, custom operators), the compiler uses CUDA 13.0 headers while linking against PyTorch's CUDA 12.8 runtime. This version conflict causes silent failures.

### How do I fix it?

**Option A: Use Docker (Recommended)**

Docker isolates the CUDA environment completely:

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.0"

RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Option B: Patch PyTorch's cpp_extension.py**

Edit `.venv/lib/python3.12/site-packages/torch/utils/cpp_extension.py` around line 546 to allow ±1 CUDA major version mismatch. This is a workaround Docker is cleaner.

---

## Challenge 3: No ARM64 Wheels for Common Packages

### Which packages are affected?

| Package | ARM64 + CUDA Wheel? | Solution |
|---------|---------------------|----------|
| mmcv | ❌ No | Build from source |
| flash_attn | ❌ No | Build from source |
| decord | ❌ No | Feature unavailable |
| xtcocotools | ❌ No | Build from source |
| spacy (pinned versions) | ⚠️ Some versions missing | Use `>=3.7.0` |

### How do I build mmcv from source?

```bash
MMCV_WITH_OPS=1 FORCE_CUDA=1 \
pip install git+https://github.com/open-mmlab/mmcv.git@v2.1.0 --no-build-isolation
```

**Build time:** Approximately 25 minutes (compiling CUDA kernels).

### How do I build Flash Attention 2 from source?

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git submodule update --init csrc/cutlass

CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST="12.0" \
FLASH_ATTN_CUDA_ARCHS="120" \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
MAX_JOBS=4 \
pip install -e . --no-build-isolation
```

**Build time:** 15-20 minutes.

### What about decord?

There's no ARM64 wheel and no straightforward build path. Features requiring decord (like some video-to-video pipelines) must be disabled or refactored to use alternative libraries.

### How do I handle pinned spacy versions?

If a project pins `spacy==3.8.4` (or similar) and no ARM64 wheel exists:

```dockerfile
RUN sed -i 's/spacy==3.8.4/spacy>=3.7.0/' /app/requirements.txt
```

---

## Challenge 4: Python 3.12 Breaks MMLab Stack

### What error will I see?

```
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
```

### Why does this happen?

Ubuntu 24.04 ships with Python 3.12 by default. The MMLab ecosystem (mmcv, mmdet, mmpose) depends on `pkg_resources`, which uses deprecated Python APIs that were removed in Python 3.12.

### How do I fix it?

Use Python 3.10 for MMLab projects:

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3-pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
```

**Note:** Non-MMLab projects work fine with Python 3.12.

---

## Challenge 5: PyTorch 2.6+ Breaks Old Checkpoints

### What error will I see?

```
_pickle.UnpicklingError: Weights only load failed... weights_only=True
```

### Why does this happen?

PyTorch 2.6+ changed `torch.load()` to use `weights_only=True` by default as a security fix (CVE-2025-32434). Old model checkpoints that contain non-tensor objects fail to load.

### How do I fix it?

Patch `torch.load` at application startup:

```python
import functools
import torch

_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
```

Add this before any model loading code runs.

---

## Challenge 6: NGC Containers Don't Support GB10

### What error will I see?

```
WARNING: Detected NVIDIA GB10 GPU, which is not yet supported
```

### How do I fix it?

Don't use NGC PyTorch containers. Use base CUDA images instead:

```dockerfile
# ❌ Don't use:
FROM nvcr.io/nvidia/pytorch:24.10-py3

# ✅ Use instead:
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
# Then install PyTorch nightly manually
```

---

## Challenge 7: Flash Attention 3 Not Available

### Can I use Flash Attention 3 on DGX Spark?

No. Flash Attention 3 (using CuTe DSL) only supports sm_90 through sm_110. The GB10's sm_121 is not supported.

### What should I use instead?

Use Flash Attention 2 built for sm_120 (binary compatible with sm_121), or fall back to PyTorch's native scaled dot-product attention:

```python
from torch.nn.functional import scaled_dot_product_attention
```

---

## Challenge 8: Understanding Unified Memory

### How is DGX Spark memory different?

Unlike discrete GPUs where CPU RAM and GPU VRAM are separate, DGX Spark uses unified memory the 128GB is shared between CPU and GPU.

| Traditional GPU | DGX Spark |
|-----------------|-----------|
| CPU offloading saves VRAM | Offloading has no benefit (same memory pool) |
| PCIe transfer overhead | No transfer overhead |
| Discrete GPU memory is faster | Memory bandwidth is shared |

### What does this mean for my workloads?

- **Don't use** `--offload_model` or CPU offloading flags they provide zero benefit
- Large models (50GB+) load easily without OOM errors
- Performance is compute-bound, not memory-bound
- Expect ~65 seconds/step for 5B parameter diffusion models

---

## Challenge 9: Ubuntu 24.04 Package Name Changes

### What error will I see?

```
E: Unable to locate package libgl1-mesa-glx
```

### How do I fix it?

Package names changed in Ubuntu 24.04. Either use Ubuntu 22.04 as your Docker base (recommended for compatibility) or update package names:

```dockerfile
# Ubuntu 22.04 (recommended)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Or for Ubuntu 24.04, use new names:
RUN apt-get install -y libgl1  # Not libgl1-mesa-glx
```

---

## Challenge 10: Docker Runtime Configuration

### What error will I see?

```
Error response from daemon: unknown or invalid runtime name: nvidia
```

### Why does this happen?

Using `runtime: nvidia` in docker-compose.yml requires the NVIDIA Container Toolkit runtime to be explicitly configured in Docker daemon settings, which may not be set up by default.

### How do I fix it?

Use the `deploy.resources.reservations` syntax instead:

```yaml
# ❌ Don't use:
runtime: nvidia

# ✅ Use instead:
services:
  myservice:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, compute, utility]
```

### How do I verify GPU access works?

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Challenge 11: peft and transformers Version Conflicts

### What error will I see?

```
ModuleNotFoundError: No module named 'transformers.modeling_layers'
```

### Why does this happen?

Projects may pin older `transformers` versions, but the latest `peft` library requires newer transformers APIs.

### How do I fix it?

Pin peft to a version compatible with your transformers:

| transformers version | Compatible peft versions |
|----------------------|--------------------------|
| 4.50.0 | 0.10.x - 0.13.x |
| 4.51+ | 0.14.x+ |

```dockerfile
# For transformers 4.50.0:
RUN pip install "peft>=0.10.0,<0.14.0"
```

---

## Quick Reference: Error → Solution Table

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `torch.cuda.is_available()` returns False | PyTorch doesn't support sm_121 | Install PyTorch nightly with CUDA 12.8 |
| `No module named 'mmcv._ext'` | CUDA version mismatch | Use Docker with CUDA 12.8 base |
| `weights_only=True` pickle error | PyTorch 2.6+ security change | Patch `torch.load` at startup |
| `GB10 GPU not yet supported` | NGC container limitation | Use base CUDA image, not NGC |
| `ImpImporter` AttributeError | Python 3.12 + old pkg_resources | Use Python 3.10 |
| `sm_121 not supported` | Flash Attention prebuilt binaries | Build Flash Attention 2 from source |
| `CUDA version mismatch` | System CUDA 13.0 vs PyTorch 12.8 | Use Docker or patch cpp_extension.py |
| `decord` import error | No ARM64 wheel exists | Disable features requiring decord |
| `unknown runtime name: nvidia` | Docker runtime not configured | Use `deploy.resources.reservations` |
| `No module named 'transformers.modeling_layers'` | peft/transformers mismatch | Pin peft to compatible version |
| `No matching distribution for spacy==X.X.X` | Pinned spacy lacks ARM64 wheel | Use `spacy>=3.7.0` |

---

## Quick Reference: Build Times on DGX Spark

| Package | Build Time | Notes |
|---------|------------|-------|
| mmcv (with CUDA ops) | ~25 minutes | Compiling CUDA kernels |
| Flash Attention 2 | ~15-20 minutes | Building for sm_120 |
| Full Docker image (MMLab stack) | ~30-35 minutes | Including all compilations |

---

## Template: Minimal venv Setup

For simple projects that don't require MMLab:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## Template: Docker Setup for MMLab Projects

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV MMCV_WITH_OPS=1

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git wget curl ffmpeg \
    libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 \
    build-essential

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Build mmcv from source
RUN pip install git+https://github.com/open-mmlab/mmcv.git@v2.1.0 --no-build-isolation
```

---

## Template: docker-compose.yml with GPU Access

```yaml
version: '3.8'

services:
  ml-app:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, compute, utility]
    volumes:
      - ./models:/app/models
    ports:
      - "7860:7860"
```

---

## Verified Working Configurations

These configurations have been tested and verified working on DGX Spark by Martim Gaspar:

### Video Generation (Diffusion Models)
- **Approach:** Python venv with PyTorch nightly
- **VRAM usage:** ~49GB for large models
- **Performance:** Working at full speed

### Lip Sync / Avatar Animation (MMLab-based)
- **Approach:** Docker with Ubuntu 22.04 + Python 3.10
- **Key requirement:** mmcv built from source
- **Status:** Fully functional

### Image/Video Generation (5B+ parameter models)
- **Approach:** Python venv with Flash Attention 2 from source
- **Performance:** ~65 seconds/step (compute-bound)
- **Note:** Some video input features disabled due to decord

### Music Generation (LoRA Training)
- **Approach:** Docker with custom Dockerfile
- **Key fixes:** spacy version flexibility, peft version pinning
- **Status:** Training and inference working

---

## Decision Tree: Which Setup Do I Need?

```
Starting a new ML project on DGX Spark?
│
├─ Does it use MMLab (mmcv, mmdet, mmpose)?
│   ├─ YES → Use Docker with Ubuntu 22.04 + Python 3.10
│   └─ NO  → Continue below
│
├─ Does it need Flash Attention?
│   ├─ YES → Build Flash Attention 2 from source (sm_120)
│   └─ NO  → PyTorch SDPA fallback works fine
│
├─ Does it need video decoding (decord)?
│   ├─ YES → Feature not available on ARM64 find alternatives
│   └─ NO  → Continue below
│
├─ Does it pin specific package versions?
│   ├─ YES → Check ARM64 wheel availability:
│   │        • spacy: use >=3.7.0 if pinned version fails
│   │        • peft: match version to transformers
│   └─ NO  → Continue below
│
├─ Using Docker?
│   ├─ YES → Use deploy.resources.reservations syntax
│   └─ NO  → Continue below
│
└─ Basic setup:
    1. python3 -m venv .venv
    2. pip install --pre torch --index-url .../nightly/cu128
    3. Install project dependencies
    4. Patch torch.load if using old checkpoints
    5. Check for ARM64 wheel issues on any failures
```

---

## Frequently Asked Questions

### Is the DGX Spark good for ML development?

Yes, once you get past the initial setup pain. The 128GB unified memory lets you run models that would require multi-GPU setups elsewhere. After configuration, it's a powerful development machine for local AI work.

### Why isn't NVIDIA providing better software support?

The GB10 is new hardware (Blackwell architecture). NVIDIA is actively working on support NGC containers and official PyTorch releases will likely add sm_121 support in future updates. This guide bridges the gap until then.

### Should I use Docker or native Python venv?

- **Use Docker** for MMLab projects, complex dependency chains, or reproducible builds
- **Use venv** for simple projects, quick experiments, or when you need faster iteration

### Can I use this guide for other Blackwell GPUs?

Partially. The PyTorch nightly + CUDA 12.8 approach should work for any Blackwell GPU. However, specific challenges around ARM64 wheels are unique to DGX Spark's aarch64 platform.

### How do I get help if I'm stuck?

1. Check the [PyTorch DGX Spark Discussion Thread](https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744)
2. Search NVIDIA Developer Forums for GB10-specific issues
3. File issues on project GitHub repos mentioning "DGX Spark" and "sm_121"

---

## External Resources

- [PyTorch DGX Spark Discussion](https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744) Community thread with ongoing solutions
- [NVIDIA PyTorch Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-02.html) Official compatibility information
- [NVIDIA Spark CUDA-X Instructions](https://build.nvidia.com/spark/cuda-x-data-science/instructions) NVIDIA's setup guide

---

## About the Author

**Martim Ramos** DevOps Lead seeking AI Infra & Agentic AI roles | Obsessed with real AI systems, not buzzwords.orchestration.

- GitHub: [github.com/martimramos](https://github.com/martimramos)
- LinkedIn: [linkedin.com/in/martimramos](https://linkedin.com/in/martimramos)

This guide was written after spending 72+ hours debugging DGX Spark issues across multiple ML projects. If it saved you time, consider starring the repo.

---

## Changelog

| Date | Changes |
|------|---------|
| January 2025 | Initial release with 11 documented challenges |
| January 2025 | Added music generation configuration, peft/transformers compatibility |
| January 2025 | Added Docker runtime configuration guidance |

---

## Contributing

Found a new issue? Discovered a better solution? Contributions welcome:

1. Open an issue describing the problem and your DGX Spark configuration
2. Submit a PR with your fix and reproduction steps
3. Help others in the discussions we're all figuring this out together

---

*This guide is maintained by Martim Ramos. Last verified on NVIDIA DGX Spark with GB10 (Blackwell), January 2025.*
