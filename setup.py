from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="knn_from_scratch",
    version="1.0.0",
    author="M. Hossein Ghaemi",
    author_email="h.ghaemi.2003@gmail.com",
    description="A simple K-Nearest Neighbors implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hghaemi/knn_from_scratch.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "examples": [
            "matplotlib>=3.0",
            "jupyter>=1.0",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "knn-demo=examples.basic_example:main",
        ],
    },
)