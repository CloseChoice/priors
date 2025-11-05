#!/usr/bin/env python3
"""Convert semantic-release version to PEP 440 and SemVer formats."""

import sys
import re

version = sys.argv[1] if len(sys.argv) > 1 else ""
format_type = sys.argv[2] if len(sys.argv) > 2 else "pep440"

if format_type == "semver":
    # Keep SemVer format: 0.1.1-rc.1 (for Rust Cargo.toml)
    print(version)
else:
    # Convert to PEP 440: 0.1.1-rc.1 -> 0.1.1.rc1 (for Python pyproject.toml)
    pep440 = re.sub(r"-(\w+)\.(\d+)", r".\1\2", version)
    print(pep440)
