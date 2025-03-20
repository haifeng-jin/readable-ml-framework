import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "framework._ops.mytensor",  # Corrected: Full path to the module
        ["framework/_ops/mytensor.cpp"],  # Path to your C++ source file
        include_dirs=[pybind11.get_include()],
        cxx_standard=11,
    ),
]

# Setup function
setuptools.setup(
    name="myframework",  # Changed:  Name of the top-level package
    version="0.0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="A custom tensor library within a framework",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    install_requires=["pybind11"],
    packages=["framework", "framework._ops"], # Added:  Declare the packages
    package_dir={"": "."}, # Added:  Map the top-level to the current directory
)

