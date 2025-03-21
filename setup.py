import setuptools
from pybind11.setup_helpers import Pybind11Extension
import pybind11

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "framework.core",  # Corrected: Full path to the module
        [
            "framework/core/tensor.cpp",
            "framework/core/python_bind.cpp"
        ],  # Path to your C++ source file
        include_dirs=[pybind11.get_include()],
    ),
]

# Setup function
setuptools.setup(
    name="readable-ml-framework",  # Changed:  Name of the top-level package
    ext_modules=ext_modules,
    packages=setuptools.find_packages(), # Added:  Declare the packages
)

