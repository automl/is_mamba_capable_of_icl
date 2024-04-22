from setuptools import setup, find_packages

setup(
    name='mamba_mod',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['mamba_ssm', 'causal-conv1d'],
)