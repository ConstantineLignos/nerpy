#! /usr/bin/env python

from setuptools import find_packages, setup


def setup_package() -> None:
    setup(
        name="nerpy",
        version="0.1.0-dev",
        packages=find_packages(include=("nerpy", "nerpy.*")),
        # Package type information
        package_data={"nerpy": ["py.typed"]},
        # 3.7 and up, but not Python 4
        python_requires="~=3.7",
        license="MIT",
        long_description="Python library for named entity recognition (NER)",
        install_requires=[
            "attrs>=19.2.0",
            "python-crfsuite>=0.9.6",
            "regex",
            "frozendict",
            "numpy",
            "quickvec",
        ],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
        ],
        project_urls={"Source": "https://github.com/ConstantineLignos/nerpy"},
    )


if __name__ == "__main__":
    setup_package()
