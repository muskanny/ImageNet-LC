from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="imagenet-lc",
    version="1.0.0",
    description="ImageNet-LC: Assessing Robustness under Localized Corruptions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Muskan Singh",
    packages=find_packages(),
    package_data={"imagenet_lc": ["data/imagenet_wnids.txt"]},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "Pillow>=9.0.0",
        "tqdm>=4.60.0",
        "ultralytics>=8.3.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "lpips>=0.1.4",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)