# Contributing to Open Polymer

Thank you for your interest in contributing to Open Polymer! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/open_polymer.git
   cd open_polymer
   ```
3. **Set up the development environment**:
   ```bash
   conda create -n polymer_pred python=3.10 -y
   conda activate polymer_pred
   conda install -c conda-forge rdkit -y
   pip install -r requirements.txt
   ```

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Include a clear description of the problem
- Provide steps to reproduce the issue
- Include error messages and logs
- Specify your environment (OS, Python version, etc.)

### Suggesting Enhancements

- Use the GitHub issue tracker with the "enhancement" label
- Clearly describe the proposed feature
- Explain why it would be useful
- Provide examples if possible

### Code Contributions

1. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards:
   - Follow PEP 8 style guide
   - Add docstrings to functions and classes
   - Include type hints where appropriate
   - Write clear commit messages

3. **Test your changes**:
   ```bash
   python src/data_preprocessing.py  # Test preprocessing
   python src/train.py              # Test full pipeline
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: Clear description of your changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Submit a Pull Request** on GitHub

## Coding Standards

### Python Style
- Follow PEP 8 conventions
- Use meaningful variable names
- Keep functions focused and small
- Add comments for complex logic

### Documentation
- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Update type hints
- Include usage examples

### Commit Messages
- Use present tense ("Add feature" not "Added feature")
- Start with a verb (Add, Fix, Update, Remove)
- Be specific and descriptive
- Reference issues when applicable (#123)

## Areas for Contribution

We welcome contributions in the following areas:

### High Priority
- [ ] Hyperparameter tuning for models
- [ ] Additional molecular descriptors
- [ ] Ensemble methods
- [ ] Unit tests and integration tests
- [ ] Performance optimizations

### Medium Priority
- [ ] Additional model architectures
- [ ] Data visualization improvements
- [ ] Documentation and tutorials
- [ ] Example notebooks
- [ ] Cross-validation utilities

### Future Enhancements
- [ ] Web interface for predictions
- [ ] SMILES augmentation techniques
- [ ] Multi-task learning improvements
- [ ] Attention visualization
- [ ] Model interpretability tools

## Testing

Before submitting a pull request:

1. **Test data processing**:
   ```python
   from src.data_preprocessing import MolecularDataProcessor
   processor = MolecularDataProcessor()
   # Test with sample SMILES
   ```

2. **Test model training**:
   ```bash
   python src/train.py
   ```

3. **Verify imports**:
   ```python
   from src.models import TraditionalMLModel, GNNModel, TransformerModel
   ```

## Code Review Process

1. Maintainers will review your PR
2. Feedback will be provided if changes are needed
3. Once approved, your PR will be merged
4. Your contribution will be acknowledged in the README

## Questions?

- Open an issue for questions
- Check existing issues first
- Be respectful and constructive

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Open Polymer! 🧪

