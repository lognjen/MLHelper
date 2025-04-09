from setuptools import setup, find_packages

setup(
    name="MLHelper",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "joblib>=1.0.0",
    ],
    author="Ognjen Maksimovic",
    author_email="ognjen.maksimovic@gmail.com",
    description="A collection of helper functions for common machine learning tasks",
    keywords="machine learning, data science, helper functions",
    url="https://github.com/lognjen/MLHelper",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)
