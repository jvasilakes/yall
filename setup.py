import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pall-jvasilakes",
    version="0.0.1",
    author="Jake Vasilakes",
    author_email="jvasilakes@gmail.com",
    description="Python Active Learning Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages() + ["pall"],
    package_data={"pall": ["data/"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
