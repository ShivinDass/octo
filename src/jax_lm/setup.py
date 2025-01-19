from setuptools import setup, find_packages

setup(
    name="metagradients",
    version="0.1.0",
    description="A minimal package for meta-gradient computations.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/metagradients",  # Update with the actual URL
    packages=find_packages(),
    install_requires=[],  # Add dependencies here, e.g., ["numpy", "torch"]
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Update with your project's license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

