from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neural-scaling-predictor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Neural Scaling Predictor for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural-scaling-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "pre-commit>=2.0",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "matplotlib>=3.0",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "nsp-train=scripts.train_model:main",
            "nsp-evaluate=scripts.evaluate_model:main",
            "nsp-predict=scripts.predict:main",
        ],
    },
)
