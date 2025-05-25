# setup.py
from setuptools import setup, find_packages

setup(
    name="noiseprobe",
    version="0.1.0",
    description="Modular robustness evaluation toolkit",
    packages=find_packages(),        # <-- this will pick up noiseprobe/ and its subpackages
    install_requires=[
        "torch",
        "numpy",
        "torchvision",
        "Pillow",
    ],
    python_requires=">=3.7",
)
