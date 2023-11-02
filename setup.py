from setuptools import find_packages
from distutils.core import setup

setup(
    name="BaseWVN",
    version="0.0.1",
    author="Jiaqi Chen",
    author_email="chenjiaq@student.ethz.ch",
    packages=find_packages(),
    python_requires=">=3.6",
    description="A small example package",
    install_requires=["kornia>=0.6.5"],
)
