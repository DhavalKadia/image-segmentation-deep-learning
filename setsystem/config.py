"""
    When GPU_ENV is True, required version of CUDA is required to be installed
    Installation website: https://developer.nvidia.com/cuda-toolkit-archive
"""
GPU_ENV = True

# required packages to install first
BASE_PACKAGES = {
    "packaging": "packaging",
    "importlib_metadata": "importlib_metadata",
    "setuptools": "setuptools"
}

# required packages to run the system
PACKAGES = {
    "numpy": "numpy",
    "skimage": "scikit-image",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "h5py": "h5py",
    "tqdm": "tqdm",
    "elasticdeform": "elasticdeform"
}

# GPU specific packages
GPU_SPECIFIC_PACKAGES = {
    "tensorflow": ["tensorflow", "tensorflow-gpu"]
}

# required packages with specific version requirement
PACKAGE_VERSION = {
    "tensorflow": "2.0.0",
    "setuptools": "40.8.0"
}
