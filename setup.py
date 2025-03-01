from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cobra-autograd",
    version="0.1.0",
    author="Amir Alasady",
    author_email="amiralasady107@gmail.com",
    description="A lightweight automatic differentiation library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmirAlasady/cobra-autograd.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.21.0',
    ],
    extras_require={
        'cuda12': ['cupy-cuda12x>=13.3.0'],
        'cuda11': ['cupy-cuda11x>=12.0.0'],
        'gpu': ['cupy-cuda12x>=13.3.0'],
        'dev': [
            'pytest>=7.0.0',
            'twine>=4.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0'
        ],
        'all': [
            'cupy-cuda12x>=13.3.0',
            'pytest>=7.0.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'cobra-check=cobra_autograd.cli:check_installation',
        ],
    },
)
