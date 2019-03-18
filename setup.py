import setuptools

# See note below for more information about classifiers
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Science/Research",
]

dependencies = ["numpy", "scipy", "pyfftw", "matplotlib", "configobj", "validate"]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resistics",
    version="0.0.1",
    description="Robust magnetotelluric processing package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Neeraj Shah",
    author_email="resistics@outlook.com",
    url="https://www.resistics.io",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=dependencies,
    classifiers=classifiers,
)

