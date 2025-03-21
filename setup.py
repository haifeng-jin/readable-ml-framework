import setuptools
from pybind11.setup_helpers import Pybind11Extension
import pybind11

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "framework.core.tensor",  # Corrected: Full path to the module
        ["framework/core/tensor.cpp"],  # Path to your C++ source file
        include_dirs=[pybind11.get_include()],
        cxx_standard=11,
    ),
]

# Setup function
setuptools.setup(
    name="readable-ml-framework",  # Changed:  Name of the top-level package
    ext_modules=ext_modules,
    packages=["framework", "framework.core"], # Added:  Declare the packages
)

