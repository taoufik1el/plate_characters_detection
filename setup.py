from setuptools import setup, find_packages


def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


setup(
    name="moroccan-licence-plate",
    version="0.1.2",
    author="Taoufik El Khaouja",
    author_email="el1khaouja.taoufik@gmail.com",
    description="Detection and recognition of moroccan car licence plates.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/taoufik1el/plate_characters_detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements(),
)
