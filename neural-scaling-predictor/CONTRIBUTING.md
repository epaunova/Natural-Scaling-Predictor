# Contributing to Neural Scaling Predictor

We welcome contributions to the Neural Scaling Predictor project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/neural-scaling-predictor.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Submit a pull request

## Development Environment

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU support)

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **isort** for import sorting
- **Type hints** for better code documentation
