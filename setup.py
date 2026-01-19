from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    README = fh.read()

setup(
    name="xplique-adapters",
    version="0.1.0",
    description="Model-specific adapters for Xplique",
    long_description=README,
    long_description_content_type="text/markdown",
    license="See LICENSE",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "xplique>=2.0.0",  # Core xplique library
    ],
    extras_require = {
        'yolo': ['ultralytics>=8.0.0'],
        'detr': ['transformers'],
        'torchvision': ['torchvision'],
        'retinanet': ['keras-cv'],
        'test': [
            'pytest>=6.0.0',
            'pytest-cov',
            'pillow',
            'matplotlib',
        ],
        'all': ['ultralytics>=8.0.0', 'transformers', 'torchvision', 'keras-cv'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
