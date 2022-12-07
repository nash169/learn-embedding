#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="learn-embedding",
    version="1.0.0",
    author="Bernardo Fichera",
    author_email="bernardo.fichera@gmail.com",
    description="Non-linear dynamical system higher dimension space deformation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nash169/learn-embedding.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",                # math
        "matplotlib",           # plotting
        "torch",                # net framework
    ],
    extras_require={
        "pytorch": [
            "torchvision",      # net framework GPU
            "tensorboard"       # visualizations
        ],
        "dev": [
            "pylint",           # python linter
        ]
    },
)
