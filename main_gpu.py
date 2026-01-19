#!/usr/bin/env python
"""Puray GPU - GPU-accelerated Pure Python Raytracer for NVIDIA Blackwell (GB10).

This version uses CUDA kernels for massively parallel ray tracing,
achieving orders of magnitude speedup over CPU multiprocessing.
"""
import argparse
import importlib
import os
import time

from gpu_engine import GPURenderEngine
from scene import Scene


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated ray tracer for NVIDIA Blackwell (GB10)"
    )
    parser.add_argument("scene", help="Path to scene file (without .py extension)")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark comparing CPU vs GPU"
    )
    args = parser.parse_args()

    mod = importlib.import_module(args.scene)
    scene = Scene(mod.CAMERA, mod.OBJECTS, mod.LIGHTS, mod.WIDTH, mod.HEIGHT)

    os.chdir(os.path.dirname(os.path.abspath(mod.__file__)))

    if args.benchmark:
        run_benchmark(scene, mod.RENDERED_IMG)
    else:
        run_gpu_render(scene, mod.RENDERED_IMG)


def run_gpu_render(scene, output_file):
    """Render using GPU acceleration."""
    engine = GPURenderEngine()

    start = time.perf_counter()
    engine.render(scene, output_file)
    elapsed = time.perf_counter() - start

    print(f"GPU render time: {elapsed:.3f}s")
    print(f"Pixels per second: {scene.width * scene.height / elapsed:,.0f}")


def run_benchmark(scene, output_file):
    """Compare CPU vs GPU rendering performance."""
    from multiprocessing import cpu_count
    from engine import RenderEngine

    print("=" * 60)
    print("BENCHMARK: CPU (multiprocessing) vs GPU (CUDA)")
    print("=" * 60)
    print(f"Resolution: {scene.width}x{scene.height}")
    print(f"Total pixels: {scene.width * scene.height:,}")
    print()

    # CPU render
    print("Running CPU render...")
    cpu_engine = RenderEngine()
    cpu_output = output_file.replace(".ppm", "_cpu.ppm")

    cpu_start = time.perf_counter()
    with open(cpu_output, "w") as f:
        cpu_engine.render_multiprocess(scene, cpu_count(), f)
    cpu_elapsed = time.perf_counter() - cpu_start

    print(f"CPU time: {cpu_elapsed:.3f}s ({cpu_count()} processes)")
    print()

    # GPU render
    print("Running GPU render...")
    gpu_engine = GPURenderEngine()
    gpu_output = output_file.replace(".ppm", "_gpu.ppm")

    gpu_start = time.perf_counter()
    gpu_engine.render(scene, gpu_output)
    gpu_elapsed = time.perf_counter() - gpu_start

    print(f"GPU time: {gpu_elapsed:.3f}s")
    print()

    # Summary
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    speedup = cpu_elapsed / gpu_elapsed if gpu_elapsed > 0 else float('inf')
    print(f"CPU: {cpu_elapsed:.3f}s ({scene.width * scene.height / cpu_elapsed:,.0f} px/s)")
    print(f"GPU: {gpu_elapsed:.3f}s ({scene.width * scene.height / gpu_elapsed:,.0f} px/s)")
    print(f"Speedup: {speedup:.1f}x")
    print()
    print(f"CPU output: {cpu_output}")
    print(f"GPU output: {gpu_output}")


if __name__ == "__main__":
    main()
