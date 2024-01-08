from setuptools import setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="proteingym",
    description="ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction",
    version="1.0",
    license="MIT",
    packages=["proteingym"]
)