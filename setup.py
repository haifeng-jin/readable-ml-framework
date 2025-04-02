import glob

import pybind11
import setuptools
from pybind11.setup_helpers import Pybind11Extension

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "framework.core",  # Corrected: Full path to the module
        glob.glob("framework/core/**/*.cpp", recursive=True),
        include_dirs=[pybind11.get_include()],
    ),
]

# Setup function
setuptools.setup(
    name="readable-ml-framework",  # Changed:  Name of the top-level package
    ext_modules=ext_modules,
    packages=setuptools.find_packages(),  # Added:  Declare the packages
)
