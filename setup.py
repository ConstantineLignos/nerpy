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
            "immutabledict",
            "numpy",
            "quickvec @ https://github.com/ConstantineLignos/quickvec/archive/fef37d56af03288cee758a2ab6f9d70cc035f0d5.zip#egg=quickvec-0.2.0-dev",
        ],
        extras_require={
            "dev": [
                "pytest",
                "pytest-cov",
                "black==19.10b0",
                "isort",
                "flake8",
                "flake8-bugbear",
                "mypy==0.770",
                "tox",
                "sphinx",
            ],
            # TODO: Add extras for sequencemodels
        },
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        project_urls={"Source": "https://github.com/ConstantineLignos/nerpy"},
    )


if __name__ == "__main__":
    setup_package()
