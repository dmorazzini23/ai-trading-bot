# Development Environment Setup

This document provides comprehensive instructions for setting up the development environment for the AI Trading Bot project.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/dmorazzini23/ai-trading-bot.git
cd ai-trading-bot

# Install dependencies
make install-dev

# Validate environment
make validate-env

# Run tests
make test-all
```

## Dependency Management

### Requirements Files

- **`requirements.txt`** - Core production dependencies
- **`requirements-dev.txt`** - Development and testing dependencies
- **`pyproject.toml`** - Modern Python packaging configuration

### Installing Dependencies

```bash
# Production dependencies only
make install

# Development dependencies (includes production)
make install-dev

# Manual installation
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Testing

### Available Test Commands

```bash
# Run all tests with validation
make test-all

# Run tests quickly (fail fast)
make test-fast

# Run tests in CI mode (fewer failures, shorter output)
make test-ci

# Run tests with coverage
make coverage

# Run benchmarks only
make benchmark

# Validate environment dependencies
make validate-env
```

### Test Configuration

Tests are configured via `pytest.ini`:
- 30-second timeout for individual tests
- Excludes slow tests by default (use `-m slow` to include)
- Parallel execution with pytest-xdist
- Warning filters to reduce noise

### Test Markers

- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.smoke` - Quick smoke tests  
- `@pytest.mark.benchmark` - Performance benchmarks
- `@pytest.mark.integration` - Integration tests

## Code Quality

### Linting and Type Checking

```bash
# Run linting
make lint

# Run type checking
make mypy-check

# Run both
make check
```

### Configuration Files

- **`.flake8`** - Flake8 linting configuration
- **`pyproject.toml`** - Modern tool configuration
- **`pytest.ini`** - Test configuration

## CI/CD Workflows

### GitHub Actions

1. **`ci.yml`** - Main CI workflow for pull requests
2. **`python.yml`** - Lint and test workflow
3. **`python-app.yml`** - Coverage testing
4. **`perf-check.yml`** - Performance benchmarks
5. **`deploy.yml`** - Production deployment

All workflows use consistent dependency installation:

```yaml
- name: Install dependencies
  run: make install-dev
```

## Environment Validation

The `scripts/validate_test_environment.py` script checks for all required dependencies:

```bash
python scripts/validate_test_environment.py
```

This script:
- ✅ Validates all testing dependencies are available
- ❌ Reports missing packages with clear error messages  
- 💡 Provides installation instructions

## Troubleshooting

### Common Issues

**"No module named pytest"**
```bash
make install-dev
```

**Tests timing out**
- Increase timeout in `pytest.ini`
- Use `make test-fast` for quicker feedback

**Import errors in tests**
- Ensure `PYTHONPATH=.` is set (handled by Makefile)
- Check that all dependencies are installed

**CI failures due to dependencies**
- All workflows now use `make install-dev` for consistency
- Check workflow logs for specific package installation failures

### Network Issues

If experiencing PyPI timeouts:
```bash
# Use a different index
pip install -i https://pypi.python.org/simple/ -r requirements-dev.txt

# Or increase timeout
pip install --timeout 60 -r requirements-dev.txt
```

## Contributing

### Before Submitting a PR

1. Install development dependencies: `make install-dev`
2. Validate environment: `make validate-env`
3. Run linting: `make lint`
4. Run tests: `make test-all`
5. Check coverage: `make coverage`

### Adding New Dependencies

1. **Production dependencies**: Add to `requirements.txt`
2. **Development dependencies**: Add to `requirements-dev.txt`
3. **Update pyproject.toml** if needed
4. **Test the change**: `make validate-env && make test-all`

## Python Version

This project requires **Python 3.12.3** exactly. This is enforced in:
- `pyproject.toml` (`requires-python = "==3.12.3"`)
- All GitHub Actions workflows
- Production deployment scripts

## Architecture Notes

This project follows the AI-only maintenance policy outlined in `AGENTS.md`:
- Dependencies are managed through requirements files, not manual installation
- CI workflows are standardized and consistent
- Testing is comprehensive with proper error handling
- Environment validation ensures reproducible setups