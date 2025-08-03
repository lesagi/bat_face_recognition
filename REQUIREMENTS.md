# Requirements Management Guide

This document explains the different requirements files and how to use them for the Bat Face Recognition project.

## 📋 Requirements Files Overview

### 1. `requirements.txt` - **Production Requirements**
Contains all necessary dependencies to run the application in production.

```bash
pip install -r requirements.txt
```

**Use when:**
- Deploying to production
- Setting up the application for end users
- You want the latest compatible versions

### 2. `requirements-exact.txt` - **Exact Version Requirements**
Contains the exact versions that are known to work together (based on the working `app/requirements.txt`).

```bash
pip install -r requirements-exact.txt
```

**Use when:**
- You want to replicate the exact working environment
- Debugging version compatibility issues
- Setting up CI/CD with reproducible builds

### 3. `requirements-dev.txt` - **Development Requirements**
Contains all development tools, testing frameworks, and additional utilities.

```bash
pip install -r requirements-dev.txt
```

**Use when:**
- Setting up a development environment
- Contributing to the project
- Running tests and code quality checks

## 🚀 Quick Start Guide

### For Users (Running the Application)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install production requirements
pip install -r requirements.txt

# Run the application
python face_annotation_eyes_nose/create_augmented_dataset.py
```

### For Developers (Contributing)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development requirements (includes production deps)
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run development checks
make dev-check
```

### For Exact Reproduction
```bash
# Use exact versions from working environment
pip install -r requirements-exact.txt
```

## 🔧 Development Tools Included

### Code Quality
- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks

### Testing
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities

### Documentation
- **sphinx**: Documentation generation
- **jupyter**: Interactive notebooks

### Development Utilities
- **rich**: Beautiful terminal output
- **watchdog**: File watching
- **memory-profiler**: Performance profiling

## 📦 Core Dependencies Explained

### Deep Learning
- **tensorflow (2.15.0)**: Main deep learning framework
- **torch (2.3.0)**: PyTorch for YOLO models
- **ultralytics (8.2.18)**: YOLO implementation

### Computer Vision
- **opencv-python (4.9.0.80)**: Image processing
- **albumentations (1.4.8)**: Data augmentation
- **Pillow (10.3.0)**: Image handling

### Data Science
- **numpy (≥1.24.4)**: Numerical computing
- **pandas (2.2.2)**: Data manipulation
- **scikit-learn (1.5.0)**: Machine learning utilities
- **matplotlib (3.8.4)**: Plotting and visualization

### Web Framework
- **Flask (3.0.2)**: Web application framework

### Experiment Tracking
- **mlflow (2.10.2)**: ML experiment management

## 🐳 Docker Usage

If you prefer using Docker (Dockerfile not included but can be created):

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "app/main.py"]
```

## 🔄 Updating Requirements

### When adding new dependencies:
1. Add to appropriate requirements file
2. Update version pins if needed
3. Test compatibility
4. Update this documentation

### When updating versions:
```bash
# Check for outdated packages
pip list --outdated

# Update requirements files
pip-compile requirements.in  # If using pip-tools
```

## 🚨 Common Issues & Solutions

### Issue: Package conflicts
**Solution:** Use exact requirements or create fresh virtual environment

### Issue: CUDA/GPU related errors
**Solution:** 
- Install CUDA-compatible versions
- Use CPU-only versions if needed
- Check PyTorch CUDA compatibility

### Issue: OpenCV import errors
**Solution:**
- Try `opencv-python-headless` instead
- Install system dependencies on Linux

### Issue: Memory errors during training
**Solution:**
- Reduce batch size
- Use gradient accumulation
- Consider cloud computing resources

## 📚 Additional Resources

- [Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [Pip Requirements Files](https://pip.pypa.io/en/stable/reference/requirements-file-format/)
- [TensorFlow Installation](https://www.tensorflow.org/install)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)

## 🤝 Contributing

When contributing:
1. Install development requirements
2. Set up pre-commit hooks
3. Run tests before submitting PR
4. Update requirements if adding dependencies

```bash
# Full development setup
make quick-start

# Run all checks
make ci
```