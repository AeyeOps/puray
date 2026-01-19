# Puray

A Pure Python Raytracer by Arun Ravindran.

Watch the video tutorial: https://www.youtube.com/watch?v=KaCe63v4D_Q&list=PL8ENypDVcs3H-TxOXOzwDyCm5f2fGXlIS

Read the blog series: https://arunrocks.com/ray-tracer-in-python-1-points-in-3d-space-show-notes/

## GPU Acceleration

The clean separation of concerns in the original code made it straightforward to add GPU acceleration. The new `gpu_engine.py` uses Numba CUDA to run ray tracing on NVIDIA GPUs.

### Performance

Tested on NVIDIA GB10 (Blackwell, Compute Capability 12.1):

| Scene | Resolution | CPU (20 cores) | GPU | Speedup |
|-------|------------|----------------|-----|---------|
| twoballs | 960x540 | 0.82s | 0.25s | 3x |
| manyballs | 1920x1080 | 14.3s | 0.32s | **45x** |

Single-core CPU baseline for manyballs: ~170s → GPU achieves **530x** speedup.

### Usage

```bash
# CPU (original)
python main.py examples.twoballs

# GPU
python main_gpu.py examples.twoballs
```

### Requirements

- NVIDIA GPU with CUDA support
- Python 3.12+
- numba >= 0.60.0
- numpy >= 1.26.0

Install with:
```bash
pip install numba numpy
```

### Key Changes

1. **Structure-of-arrays memory layout** - Scene data packed for coalesced GPU memory access
2. **Iterative ray tracing** - Replaced recursion with iteration (required for CUDA)
3. **Binary PPM output (P6)** - Eliminated the original I/O bottleneck (was 96% of runtime)
4. **fastmath compilation** - Enabled fast floating-point operations in CUDA kernels

The GPU implementation preserves the original's clarity while achieving real-time performance on modern hardware.
