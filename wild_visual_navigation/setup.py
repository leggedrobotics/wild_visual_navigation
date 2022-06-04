from setuptools import find_packages
from distutils.core import setup

setup(
    name="template_project_name",
    version="0.0.1",
    author="Jonas Frey, Matias Mattamala",
    author_email="jonfrey@ethz.ch, matias@leggedrobotics.com",
    packages=find_packages(),
    python_requires=">=3.6",
    description="A small example package",
    install_requires=[""],
)