from setuptools import setup, find_packages

setup(
    name="cobra-autograd",
    version="0.1.0",
    author="Amir Alasady",
    author_email="amiralasady107@gmail.com",
    description="A lightweight automatic differentiation library",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AmirAlasady/cobra-autograd.git",  # Update with your actual repo URL
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
    python_requires='>=3.10',  # More compatible version range
    install_requires=[
        'numpy>=1.21.0',  # Updated to match CuPy's requirements
    ],
    extras_require={
        # GPU variants
        'cuda12': ['cupy-cuda12x>=13.3.0'],
        'cuda11': ['cupy-cuda11x>=12.0.0'],
        
        # Default GPU extra (CUDA 12)
        'gpu': ['cupy-cuda12x>=13.3.0'],
        
        # Development tools
        'dev': [
            'pytest>=7.0.0',
            'twine>=4.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0'
        ],
        
        # Full installation
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