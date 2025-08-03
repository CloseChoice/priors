# TreeSHAP-RS

A high-performance Rust implementation of TreeSHAP with Python bindings.

This is under active development and far ahead from a first official release.
Currently we only support `predict_proba` for decision trees.

Here, we use [`maturin`][maturin] for building Python wheels and
[`nox`][nox] for managing Python dependencies and virtualenvs.

## Development Build

For development and testing, use the debug build which compiles faster but runs slower:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode (debug build)
maturin develop
```

This allows you to make changes to the Rust code and quickly rebuild for testing.

## Performance Build

For benchmarking and production use, build with release optimizations:

```bash
# Build with maximum optimizations
maturin develop --release
```

Or alternatively, build a wheel and install it:

```bash
# Build optimized wheel
maturin build --release

# Install the optimized wheel
pip install target/wheels/*.whl --force-reinstall
```

The release build includes these optimizations (configured in `Cargo.toml`):
- Maximum optimization level (`opt-level = 3`)
- Link-time optimization (`lto = true`) 
- Single codegen unit for better optimization
- Stripped debug symbols for smaller binaries

## Testing

Running `nox` inside this directory creates a virtualenv,
installs Python dependencies and the extension into it
and executes the tests from `tests/test_treeshap.py`.

## Usage

From inside a virtualenv with the extension installed:

```python
>>> import numpy as np
>>> from treeshap import predict_proba
>>> # Your TreeSHAP code here
```

[maturin]: https://github.com/PyO3/maturin
[nox]: https://github.com/theacodes/nox
