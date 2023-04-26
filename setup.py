from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages, setup

description = "MC Tuning Tool based on Jax"

setup(
    name="mctuner",
    version="0.1.0",
    description=description,
    author="Xiangyang Ju",
    author_email="xiangyang.ju@gmail.com",
    url="https://github.com/HEPonHPC/jax_mc_tune",
    license="MIT",
    packages=find_packages(),
    install_requires=[
    ],
    extras_require={},
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    scripts=[
    ],
)
