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

# Default validation after a change (changed-file lint/type/compile checks,
# mapped regression tests, forbidden-pattern scan, runtime smoke checks)
bash scripts/agent_validate_changed.sh
```

See AGENTS.md section 4 for the variants (`--market-hours`, `--full`,
`--docs-only`). Do not run the broad test suite during market hours without
approval.

## Dependency Management

### Requirements Files

- **`requirements.txt`** - Core production dependencies
- **`requirements-dev.txt`** - Development and testing dependencies
- **`pyproject.toml`** - Modern Python packaging configuration

### Installing Dependencies

```bash
# Development dependencies (creates venv/ if missing, installs
# requirements-dev.txt and requirements-test.txt, then installs the package)
make install-dev

# Production and dev requirements into $(PYTHON): the repo venv if present, otherwise python3
make ensure-runtime

# Manual installation
python -m pip install -U pip
pip install -e .
pip install -r requirements-dev.txt
```

## Testing

### Available Test Commands

```bash
# Default validation after a change
bash scripts/agent_validate_changed.sh

# Full suite without slow, integration or credentials-marked tests
# (installs requirements first; skip with SKIP_INSTALL=1)
make test-all

# Everything, including slow/integration/credentials-marked tests
make test-all-heavy

# Targeted smoke subset
make smoke

# Validate environment dependencies
make validate-env
```

### Test Configuration

Tests are configured via `pytest.ini`:
- No per-test timeout (`timeout = 0`); `faulthandler_timeout = 120` dumps stacks for hung tests
- `addopts = -ra`; tests run serially (the xdist preload was dropped)
- `filterwarnings = error`, with a few ignores
- `make test-all` additionally deselects slow, integration and credentials-marked tests with `-m`

### Test Markers

Listed in `pyproject.toml`. pytest reads `pytest.ini`, which declares no
markers, so these are not registered:

- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Tests that need network or vendor APIs
- `@pytest.mark.no_test_credentials` - Run without globally injected Alpaca test credentials

## Code Quality

### Linting and Type Checking

```bash
# Run linting
make lint

# Type checking
./venv/bin/mypy
bash scripts/typecheck_strict.sh
```

### Configuration Files

- **`.flake8`** - Flake8 linting configuration
- **`pyproject.toml`** - Modern tool configuration
- **`pytest.ini`** - Test configuration

## CI/CD Workflows

### GitHub Actions

1. **`ci.yml`** - Main CI workflow (push/PR + scheduled nightly full-deps checks)
2. **`codeql.yml`** - Static security analysis
3. **`workflow-lint.yml`** - Lints GitHub workflow files with actionlint
4. **`dependency-audit.yml`** - Dependency vulnerability auditing
5. **`sbom.yml`** - CycloneDX SBOM generation
6. **`scorecard.yml`** - OSSF scorecard analysis
7. **`release.yml`** - Manual tagged release build and GitHub release publishing

All workflows use consistent dependency installation:

```yaml
- name: Install dependencies
  run: pip install -r requirements-dev.txt
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
- Tests have no default timeout (`timeout = 0` in `pytest.ini`); run the
  affected test file directly, or use `bash scripts/agent_validate_changed.sh`
  for the mapped regression tests

**Import errors in tests**
- Ensure `PYTHONPATH=.` is set (handled by Makefile)
- Check that all dependencies are installed
- Keep Makefile targets side-effect free at parse time; create artifact
  directories inside the targets that write reports.

**CI failures due to dependencies**
- CI and security workflows install dependencies from `requirements*.txt`
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
4. Run the default validation: `bash scripts/agent_validate_changed.sh`
   (use `--full` for explicit full validation outside market hours)

### Adding New Dependencies

1. **Production dependencies**: Add to `requirements.txt`
2. **Development dependencies**: Add to `requirements-dev.txt`
3. **Update pyproject.toml** if needed
4. **Test the change**: `make validate-env && bash scripts/agent_validate_changed.sh`

## Python Version

This project requires Python 3.12. This is enforced in:
- `pyproject.toml` (`requires-python = ">=3.12,<3.13"`)
- All GitHub Actions workflows
- Production deployment scripts

## Architecture Notes

This project follows the AI-only maintenance policy outlined in `AGENTS.md`:
- Dependencies are managed through requirements files, not manual installation
- CI workflows are standardized and consistent
- Testing is comprehensive with proper error handling
- Environment validation ensures reproducible setups
