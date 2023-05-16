import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='merge_by_lev',
    version='0.0.3',
    author='J. A. Moreno-Guerra',
    author_email='jzs.gm27@gmail.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jzsmoreno/merge_by_lev',
    project_urls = {
        "Bug Tracker": "https://github.com/jzsmoreno/merge_by_lev"
    },
    license='MIT',
    packages=['merge_by_lev'],
    install_requires=[
        'numpy',
        'pandas'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)