"""GPU-accelerated ray tracing engine for NVIDIA Blackwell (GB10) architecture.

Uses Numba CUDA for custom kernels optimized for compute capability 12.1.
Implements iterative ray tracing with structure-of-arrays memory layout
for coalesced GPU memory access.
"""
import math
import numpy as np
from numba import cuda

# Constants for ray tracing
MAX_DEPTH = 5
MIN_DISPLACE = 0.0001
SPECULAR_K = 50

# Material type constants
MATERIAL_SOLID = 0
MATERIAL_CHEQUERED = 1


def check_gpu():
    """Check GPU availability and return device info."""
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU acceleration requires NVIDIA GPU.")

    device = cuda.get_current_device()
    cc = device.compute_capability

    print(f"GPU: {device.name}")
    print(f"Compute Capability: {cc[0]}.{cc[1]}")
    print(f"Multiprocessors: {device.MULTIPROCESSOR_COUNT}")
    print(f"Max threads per block: {device.MAX_THREADS_PER_BLOCK}")

    if cc[0] < 12:
        print(f"Warning: Compute capability {cc[0]}.{cc[1]} detected. "
              f"GB10 Blackwell (12.1) recommended for optimal performance.")

    return device


def get_optimal_block_size(device):
    """Calculate optimal block size for the device.

    For Blackwell architecture (GB10), 16x16 = 256 threads provides
    good balance between occupancy and register usage for ray tracing.
    Larger blocks (32x32) exceed register limits for this kernel.
    """
    # 16x16 = 256 threads balances occupancy with register pressure
    # Ray tracing kernels use many registers, so smaller blocks work better
    return (16, 16)


class GPUScene:
    """Scene data packed into GPU-friendly structure-of-arrays format."""

    def __init__(self, scene):
        """Convert CPU scene objects to GPU arrays.

        Args:
            scene: CPU Scene object with camera, objects, lights
        """
        self.width = scene.width
        self.height = scene.height
        self.camera = np.array([scene.camera.x, scene.camera.y, scene.camera.z],
                                dtype=np.float32)

        # Pack sphere data: center(3) + radius(1) = 4 floats per sphere
        num_spheres = len(scene.objects)
        self.sphere_centers = np.zeros((num_spheres, 3), dtype=np.float32)
        self.sphere_radii = np.zeros(num_spheres, dtype=np.float32)

        # Material data: type(1) + colors(6) + properties(4) = 11 floats per material
        self.material_types = np.zeros(num_spheres, dtype=np.int32)
        self.material_color1 = np.zeros((num_spheres, 3), dtype=np.float32)
        self.material_color2 = np.zeros((num_spheres, 3), dtype=np.float32)
        self.material_props = np.zeros((num_spheres, 4), dtype=np.float32)  # ambient, diffuse, specular, reflection

        for i, obj in enumerate(scene.objects):
            self.sphere_centers[i] = [obj.center.x, obj.center.y, obj.center.z]
            self.sphere_radii[i] = obj.radius

            mat = obj.material
            if hasattr(mat, 'color1'):  # ChequeredMaterial
                self.material_types[i] = MATERIAL_CHEQUERED
                self.material_color1[i] = [mat.color1.x, mat.color1.y, mat.color1.z]
                self.material_color2[i] = [mat.color2.x, mat.color2.y, mat.color2.z]
            else:  # Solid Material
                self.material_types[i] = MATERIAL_SOLID
                self.material_color1[i] = [mat.color.x, mat.color.y, mat.color.z]
                self.material_color2[i] = [0, 0, 0]  # unused

            self.material_props[i] = [mat.ambient, mat.diffuse, mat.specular, mat.reflection]

        # Pack light data
        num_lights = len(scene.lights)
        self.light_positions = np.zeros((num_lights, 3), dtype=np.float32)
        self.light_colors = np.zeros((num_lights, 3), dtype=np.float32)

        for i, light in enumerate(scene.lights):
            self.light_positions[i] = [light.position.x, light.position.y, light.position.z]
            self.light_colors[i] = [light.color.x, light.color.y, light.color.z]

        self.num_spheres = num_spheres
        self.num_lights = num_lights

    def to_device(self):
        """Transfer all scene data to GPU memory."""
        return GPUSceneDevice(
            cuda.to_device(self.camera),
            cuda.to_device(self.sphere_centers),
            cuda.to_device(self.sphere_radii),
            cuda.to_device(self.material_types),
            cuda.to_device(self.material_color1),
            cuda.to_device(self.material_color2),
            cuda.to_device(self.material_props),
            cuda.to_device(self.light_positions),
            cuda.to_device(self.light_colors),
            self.num_spheres,
            self.num_lights,
            self.width,
            self.height
        )


class GPUSceneDevice:
    """Container for GPU device arrays."""

    def __init__(self, camera, sphere_centers, sphere_radii, material_types,
                 material_color1, material_color2, material_props,
                 light_positions, light_colors, num_spheres, num_lights,
                 width, height):
        self.camera = camera
        self.sphere_centers = sphere_centers
        self.sphere_radii = sphere_radii
        self.material_types = material_types
        self.material_color1 = material_color1
        self.material_color2 = material_color2
        self.material_props = material_props
        self.light_positions = light_positions
        self.light_colors = light_colors
        self.num_spheres = num_spheres
        self.num_lights = num_lights
        self.width = width
        self.height = height


# CUDA device functions with fastmath for Blackwell optimization
@cuda.jit(device=True, fastmath=True)
def dot3(a0, a1, a2, b0, b1, b2):
    """Dot product of two 3D vectors."""
    return a0 * b0 + a1 * b1 + a2 * b2


@cuda.jit(device=True, fastmath=True)
def length3(x, y, z):
    """Length of a 3D vector."""
    return math.sqrt(x * x + y * y + z * z)


@cuda.jit(device=True, fastmath=True)
def normalize3(x, y, z):
    """Normalize a 3D vector, returns tuple (nx, ny, nz)."""
    mag = length3(x, y, z)
    if mag > 0:
        return x / mag, y / mag, z / mag
    return 0.0, 0.0, 0.0


@cuda.jit(device=True, fastmath=True)
def intersect_sphere(ray_ox, ray_oy, ray_oz, ray_dx, ray_dy, ray_dz,
                     center_x, center_y, center_z, radius):
    """Ray-sphere intersection test.

    Returns distance to intersection or -1 if no hit.
    """
    # Vector from ray origin to sphere center
    oc_x = ray_ox - center_x
    oc_y = ray_oy - center_y
    oc_z = ray_oz - center_z

    # Quadratic formula coefficients (a=1 since direction is normalized)
    b = 2.0 * dot3(ray_dx, ray_dy, ray_dz, oc_x, oc_y, oc_z)
    c = dot3(oc_x, oc_y, oc_z, oc_x, oc_y, oc_z) - radius * radius

    discriminant = b * b - 4.0 * c

    if discriminant >= 0:
        dist = (-b - math.sqrt(discriminant)) / 2.0
        if dist > 0:
            return dist
    return -1.0


@cuda.jit(device=True, fastmath=True)
def get_material_color(mat_type, color1, color2, pos_x, pos_y, pos_z):
    """Get material color at position (handles solid and chequered)."""
    if mat_type == MATERIAL_CHEQUERED:
        # Chequered pattern based on position
        check_x = int((pos_x + 5.0) * 3.0) % 2
        check_z = int(pos_z * 3.0) % 2
        if check_x == check_z:
            return color1[0], color1[1], color1[2]
        else:
            return color2[0], color2[1], color2[2]
    else:
        return color1[0], color1[1], color1[2]


@cuda.jit(fastmath=True)
def ray_trace_kernel(output, camera, sphere_centers, sphere_radii,
                     material_types, material_color1, material_color2,
                     material_props, light_positions, light_colors,
                     num_spheres, num_lights, width, height):
    """Main ray tracing CUDA kernel - one thread per pixel.

    Implements iterative ray tracing with reflection support.
    Uses structure-of-arrays for coalesced memory access.
    """
    # Calculate pixel coordinates
    px = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    py = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if px >= width or py >= height:
        return

    # Calculate ray direction for this pixel
    aspect_ratio = float(width) / float(height)
    x0, x1 = -1.0, 1.0
    y0, y1 = -1.0 / aspect_ratio, 1.0 / aspect_ratio

    x = x0 + (x1 - x0) * px / (width - 1)
    y = y0 + (y1 - y0) * py / (height - 1)

    # Initial ray from camera
    ray_ox = camera[0]
    ray_oy = camera[1]
    ray_oz = camera[2]

    ray_dx = x - ray_ox
    ray_dy = y - ray_oy
    ray_dz = 0.0 - ray_oz  # Target z=0 plane initially, then normalize
    ray_dx, ray_dy, ray_dz = normalize3(ray_dx, ray_dy, ray_dz)

    # Accumulated color
    final_r, final_g, final_b = 0.0, 0.0, 0.0

    # Reflection attenuation
    attenuation = 1.0

    # Iterative ray tracing (replaces recursion for GPU compatibility)
    for depth in range(MAX_DEPTH):
        # Find nearest intersection
        min_dist = 1e20
        hit_idx = -1

        for i in range(num_spheres):
            dist = intersect_sphere(
                ray_ox, ray_oy, ray_oz,
                ray_dx, ray_dy, ray_dz,
                sphere_centers[i, 0], sphere_centers[i, 1], sphere_centers[i, 2],
                sphere_radii[i]
            )
            if dist > 0 and dist < min_dist:
                min_dist = dist
                hit_idx = i

        if hit_idx < 0:
            break  # No hit, ray escapes scene

        # Calculate hit position
        hit_x = ray_ox + ray_dx * min_dist
        hit_y = ray_oy + ray_dy * min_dist
        hit_z = ray_oz + ray_dz * min_dist

        # Calculate surface normal
        center_x = sphere_centers[hit_idx, 0]
        center_y = sphere_centers[hit_idx, 1]
        center_z = sphere_centers[hit_idx, 2]
        normal_x, normal_y, normal_z = normalize3(
            hit_x - center_x, hit_y - center_y, hit_z - center_z
        )

        # Get material properties
        mat_type = material_types[hit_idx]
        color1 = material_color1[hit_idx]
        color2 = material_color2[hit_idx]
        props = material_props[hit_idx]
        ambient, diffuse, specular, reflection = props[0], props[1], props[2], props[3]

        # Get object color at hit position
        obj_r, obj_g, obj_b = get_material_color(mat_type, color1, color2, hit_x, hit_y, hit_z)

        # Ambient contribution
        color_r = ambient * 1.0  # White ambient light
        color_g = ambient * 1.0
        color_b = ambient * 1.0

        # Vector to camera
        to_cam_x = camera[0] - hit_x
        to_cam_y = camera[1] - hit_y
        to_cam_z = camera[2] - hit_z
        to_cam_x, to_cam_y, to_cam_z = normalize3(to_cam_x, to_cam_y, to_cam_z)

        # Light contributions
        for li in range(num_lights):
            light_x = light_positions[li, 0]
            light_y = light_positions[li, 1]
            light_z = light_positions[li, 2]
            light_r = light_colors[li, 0]
            light_g = light_colors[li, 1]
            light_b = light_colors[li, 2]

            # Vector to light
            to_light_x = light_x - hit_x
            to_light_y = light_y - hit_y
            to_light_z = light_z - hit_z
            to_light_x, to_light_y, to_light_z = normalize3(to_light_x, to_light_y, to_light_z)

            # Diffuse (Lambert)
            ndotl = max(0.0, dot3(normal_x, normal_y, normal_z,
                                  to_light_x, to_light_y, to_light_z))
            color_r += obj_r * diffuse * ndotl
            color_g += obj_g * diffuse * ndotl
            color_b += obj_b * diffuse * ndotl

            # Specular (Blinn-Phong)
            half_x = to_light_x + to_cam_x
            half_y = to_light_y + to_cam_y
            half_z = to_light_z + to_cam_z
            half_x, half_y, half_z = normalize3(half_x, half_y, half_z)

            ndoth = max(0.0, dot3(normal_x, normal_y, normal_z,
                                   half_x, half_y, half_z))
            spec = ndoth ** SPECULAR_K
            color_r += light_r * specular * spec
            color_g += light_g * specular * spec
            color_b += light_b * specular * spec

        # Add to final color with attenuation
        final_r += color_r * attenuation
        final_g += color_g * attenuation
        final_b += color_b * attenuation

        # Prepare reflection ray for next iteration
        if reflection > 0 and depth < MAX_DEPTH - 1:
            # Reflect ray direction around normal
            dot_dn = dot3(ray_dx, ray_dy, ray_dz, normal_x, normal_y, normal_z)
            ray_dx = ray_dx - 2.0 * dot_dn * normal_x
            ray_dy = ray_dy - 2.0 * dot_dn * normal_y
            ray_dz = ray_dz - 2.0 * dot_dn * normal_z

            # Offset origin to avoid self-intersection
            ray_ox = hit_x + normal_x * MIN_DISPLACE
            ray_oy = hit_y + normal_y * MIN_DISPLACE
            ray_oz = hit_z + normal_z * MIN_DISPLACE

            # Attenuate by reflection coefficient
            attenuation *= reflection
        else:
            break

    # Write output (clamp to [0, 1])
    output_idx = (py * width + px) * 3
    output[output_idx] = min(1.0, max(0.0, final_r))
    output[output_idx + 1] = min(1.0, max(0.0, final_g))
    output[output_idx + 2] = min(1.0, max(0.0, final_b))


class GPURenderEngine:
    """GPU-accelerated ray tracing engine for NVIDIA Blackwell architecture."""

    def __init__(self):
        """Initialize GPU render engine and check device capabilities."""
        self.device = check_gpu()
        self.block_size = get_optimal_block_size(self.device)
        print(f"Block size: {self.block_size[0]}x{self.block_size[1]}")

    def render(self, scene, output_file):
        """Render scene to file using GPU acceleration.

        Args:
            scene: Scene object with camera, objects, lights, width, height
            output_file: Path to output PPM file
        """
        width = scene.width
        height = scene.height
        total_pixels = width * height

        print(f"Rendering {width}x{height} = {total_pixels:,} pixels on GPU...")

        # Convert scene to GPU format and transfer to device
        gpu_scene = GPUScene(scene)
        device_scene = gpu_scene.to_device()

        # Allocate output buffer on GPU
        output_device = cuda.device_array(width * height * 3, dtype=np.float32)

        # Calculate grid dimensions
        grid_x = (width + self.block_size[0] - 1) // self.block_size[0]
        grid_y = (height + self.block_size[1] - 1) // self.block_size[1]
        grid_size = (grid_x, grid_y)

        print(f"Grid size: {grid_x}x{grid_y} = {grid_x * grid_y:,} blocks")
        print(f"Total threads: {grid_x * self.block_size[0] * grid_y * self.block_size[1]:,}")

        # Launch kernel
        cuda.synchronize()

        ray_trace_kernel[grid_size, self.block_size](
            output_device,
            device_scene.camera,
            device_scene.sphere_centers,
            device_scene.sphere_radii,
            device_scene.material_types,
            device_scene.material_color1,
            device_scene.material_color2,
            device_scene.material_props,
            device_scene.light_positions,
            device_scene.light_colors,
            device_scene.num_spheres,
            device_scene.num_lights,
            width, height
        )

        cuda.synchronize()
        print("GPU kernel completed.")

        # Copy result back to host
        output_host = output_device.copy_to_host()

        # Write PPM file
        self._write_ppm(output_file, output_host, width, height)
        print(f"Output written to {output_file}")

    def _write_ppm(self, filepath, pixels, width, height):
        """Write pixel data to PPM file using binary P6 format.

        Binary PPM is ~100x faster than text P3 format due to:
        - No string formatting per pixel
        - Direct binary write via NumPy
        - Single I/O operation instead of millions
        """
        # Reshape and convert to uint8 in one vectorized operation
        rgb = (np.clip(pixels.reshape(height, width, 3) * 255, 0, 255)).astype(np.uint8)

        with open(filepath, 'wb') as f:
            # P6 is binary PPM format
            f.write(f"P6\n{width} {height}\n255\n".encode())
            rgb.tofile(f)
