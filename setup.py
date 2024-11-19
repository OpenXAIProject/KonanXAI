from setuptools import find_packages, setup
with open ('./requirements.txt') as f:
    required = f.read().splitlines()
setup(
    name = "konan-xai",
    version = "2.0",
    author= "konan Tecnology Inc.",
    author_email="konan@konantech.com",
    description="konan Technology XAI",
    packages=find_packages(exclude=["*project*"]),
    install_requires = required
)
