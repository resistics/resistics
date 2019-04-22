import setuptools
import resistics

# options
ver = resistics.__version__

# See note below for more information about classifiers
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Science/Research",
]

with open("README.md", "r") as fh:
    long_description = fh.read()

project_urls = {
    "Source": "https://github.com/resistics/resistics",
    "Tracker": "https://github.com/resistics/resistics/issues",
}

dependencies = ["numpy", "scipy", "pyfftw", "matplotlib", "configobj"]

setuptools.setup(
    name="resistics",
    version=ver,
    description="Robust magnetotelluric processing package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="geophysics magnetotellurics electromagnetic resistivity statistics",
    author="Neeraj Shah",
    author_email="resistics@outlook.com",
    url="https://www.resistics.io",
    project_urls=project_urls,
    license="MIT",
    python_requires=">=3.6",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=dependencies,
    classifiers=classifiers,
)

