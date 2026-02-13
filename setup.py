"""
Setup script for WAV2HYP package.

This file provides a fallback setup method for older Python environments
that don't support pyproject.toml. The main configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages
import pathlib

# Read the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

# Read requirements
def read_requirements(filename):
    """Read requirements from file, filtering out comments and empty lines."""
    try:
        with open(filename, 'r') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Remove inline comments
                    req = line.split(';')[0].strip()
                    if req:
                        requirements.append(req)
            return requirements
    except FileNotFoundError:
        return []

requirements = read_requirements('requirements.txt')

setup(
    name="wav2hyp",
    version="1.0.0",
    description="Waveform to Hypocenter processing pipeline for seismic data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="WAV2HYP Development Team",
    author_email="support@example.com",
    url="https://github.com/your-org/wav2hyp",
    project_urls={
        "Documentation": "https://wav2hyp.readthedocs.io/",
        "Source": "https://github.com/your-org/wav2hyp",
        "Tracker": "https://github.com/your-org/wav2hyp/issues",
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "mapping": ["cartopy>=0.20.0"],
        "jupyter": ["jupyter>=1.0.0", "ipython>=7.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0", 
            "flake8>=3.9.0"
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "wav2hyp=wav2hyp.cli:cli_main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    keywords="seismology earthquake phase-picking event-association earthquake-location machine-learning volcano-monitoring",
)
