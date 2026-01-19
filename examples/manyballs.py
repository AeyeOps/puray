"""High-resolution scene with many spheres to demonstrate GPU acceleration."""
from color import Color
from light import Light
from material import ChequeredMaterial, Material
from point import Point
from sphere import Sphere
from vector import Vector
import random

# Higher resolution for GPU benchmark
WIDTH = 1920
HEIGHT = 1080
RENDERED_IMG = "manyballs.ppm"
CAMERA = Vector(0, -0.35, -1)

# Seed for reproducibility
random.seed(42)

# Generate many spheres
OBJECTS = [
    # Ground Plane
    Sphere(
        Point(0, 10000.5, 1),
        10000.0,
        ChequeredMaterial(
            color1=Color.from_hex("#420500"),
            color2=Color.from_hex("#e6b87d"),
            ambient=0.2,
            reflection=0.2,
        ),
    ),
]

# Add 50 random spheres
colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
          "#FF8800", "#8800FF", "#00FF88", "#FF0088", "#88FF00", "#0088FF"]

for i in range(50):
    x = random.uniform(-3, 3)
    z = random.uniform(0.5, 8)
    y = random.uniform(-0.3, 0.1)
    radius = random.uniform(0.15, 0.4)
    color = random.choice(colors)
    reflection = random.uniform(0.2, 0.7)

    OBJECTS.append(
        Sphere(
            Point(x, y, z),
            radius,
            Material(Color.from_hex(color), reflection=reflection)
        )
    )

LIGHTS = [
    Light(Point(1.5, -0.5, -10), Color.from_hex("#FFFFFF")),
    Light(Point(-0.5, -10.5, 0), Color.from_hex("#E6E6E6")),
    Light(Point(-3, -2, 5), Color.from_hex("#AAAAFF")),
]
