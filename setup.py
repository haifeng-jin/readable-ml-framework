import glob

import pybind11
import setuptools
from pybind11.setup_helpers import Pybind11Extension

# Define the extension module
ext_modules = [
    Pybind11Extension(
        # The Python import path to the C++ module.
        "framework.core",
        # Pass in all the C++ files needed to compile the module. We put all
        # the .cpp and .h files under framework/core dir.
        glob.glob("framework/core/**/*.cpp", recursive=True),
        # Tell setuptools where to find the pybind11 header files while
        # compiling the C++ extension module.
        include_dirs=[pybind11.get_include()],
    ),
]

# Setup function
setuptools.setup(
    # Name of the top-level package
    name="readable-ml-framework",
    # Add the C++ extension module to the package.
    ext_modules=ext_modules,
    # Let setuptools find the Python package in the current dir.
    packages=setuptools.find_packages(),
    # Specify the version of the package.
    version="0.0.1",
)
