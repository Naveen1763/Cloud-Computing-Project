# Contributing to AAPA Simulator

We welcome contributions to the AAPA project! This document provides guidelines for contributing.

## How to Contribute

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** and ensure tests pass
3. **Write clear commit messages** describing your changes
4. **Submit a pull request** with a detailed description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/aapa-simulator.git
cd aapa-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -r requirements.txt
pip install -e .
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Run `black` for code formatting:
  ```bash
  black src/ examples/ tests/
  ```

## Testing

- Write tests for new functionality
- Ensure all tests pass:
  ```bash
  pytest tests/
  ```
- Maintain test coverage above 80%

## Documentation

- Update README.md if adding new features
- Add docstrings following Google style
- Include examples for new functionality

## Submitting Issues

When submitting issues, please include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)

## Pull Request Process

1. Update documentation
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive criticism
- Help others learn and grow

Thank you for contributing to AAPA!