from setuptools import find_packages
from distutils.core import setup

# Minimum dependencies required prior to installation
# TODO: If we add the "opencv-python>=4.6", it wont build on the jetson

INSTALL_REQUIRES = [
    # generic
    "numpy",
    "tqdm",
    "kornia>=0.6.5",
    "pip",
    "torchvision",
    "torch>=1.21",
    "torchmetrics",
    "pytorch_lightning>=1.6.5",
    "pytest",
    "scipy",
    "scikit-image",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pandas",
    "pytictac",
    "torch_geometric",
    "omegaconf",
    "optuna",
    "neptune",
    "fast-slic",
    "hydra-core",
    "prettytable",
    "termcolor",
    "pydensecrf@git+https://github.com/lucasb-eyer/pydensecrf.git",
    "liegroups@git+https://github.com/mmattamala/liegroups",
    "wget",
    "rospkg",
    "wandb",
    "opencv-python==4.2.0.34",
]
setup(
    name="wild_visual_navigation",
    version="0.0.1",
    author="Jonas Frey, Matias Mattamala",
    author_email="jonfrey@ethz.ch, matias@robots.oex.ac.uk",
    packages=find_packages(),
    python_requires=">=3.7",
    description="A small example package",
    install_requires=[INSTALL_REQUIRES],
    dependencies=["https://download.pytorch.org/whl/torch-2.1.0+cu121-cp38-cp38-linux_x86_64.whl"],
    dependency_links=["https://download.pytorch.org/whl/torch-2.1.0+cu121-cp38-cp38-linux_x86_64.whl"],
)
