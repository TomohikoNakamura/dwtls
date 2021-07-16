#!/usr/bin/env python3
# Copyright (c) Tomohiko Nakamura
# All rights reserved.

from setuptools import setup, find_packages
from pathlib import Path

requirements = []
with open(Path(f"requirements.txt"), "r", encoding="utf-8") as fp:
    for line in fp:
        line = line.strip()
        if line[0] != "#":
            requirements.append(line)

setup(
    name="dwtls",
    version="1.0.2",
    url="https://github.com/TomohikoNakamura/dwtls",
    author="Tomohiko Nakamura",
    author_email="tomohiko.nakamura.jp@ieee.org",
    description="A library of trainable and fixed discrete wavelet transform layers",
    long_description=open("README.md", "rb").read().decode("utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ]
)
