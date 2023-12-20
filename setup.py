from setuptools import setup, find_packages

setup(
    name="dolos",
    version="0.0.1",
    url="https://github.com/dolos.git",
    author="Dan Oneață",
    author_email="dan.oneata@gmail.com",
    description="Weakly-supervised deepfake localization",
    packages=find_packages(),
    install_requires=[
        "click",
        "dominate",
        "grad-cam",
        "matplotlib",
        "pytorch-ignite",
        "scikit-image",
        "scikit-learn",
        "streamlit",
        "timm",
        "torchdata==0.5.1",
        "tqdm",
    ],
)
