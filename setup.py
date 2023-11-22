from setuptools import find_packages, setup

setup(
    name="distributed_catboost",
    packages=find_packages(where=".", include="distributed_catboost*"),
    version="1.0",
    author="PhysicsACE",
    description="A Ray backend for distributed Catboost",
    license="Apache 2.0",
    long_description="A distributed backend for Catboost built on top of "
    "distributed computing framework Ray.",
    url="https://github.com/PhysicsACE/distributed_catboost",
    install_requires=[
        "ray>=2.0",
        "numpy>=1.16",
        "pandas",
        "wrapt>=1.12.1",
        "xgboost>=0.90",
        "packaging",
        "catboost",
    ],
)