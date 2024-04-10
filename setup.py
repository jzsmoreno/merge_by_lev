from pathlib import Path

import setuptools

# Parse the requirements.txt file
with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "merge_by_lev"
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version

setuptools.setup(
    name="merge_by_lev",
    version=about["__version__"],
    author="J. A. Moreno-Guerra",
    author_email="jzs.gm27@gmail.com",
    description="Testing installation of Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jzsmoreno/merge_by_lev",
    project_urls={"Bug Tracker": "https://github.com/jzsmoreno/merge_by_lev"},
    license="MIT",
    packages=["merge_by_lev"],
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
