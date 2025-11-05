#!/usr/bin/env python3
"""Convert semantic-release version to PEP 440 format."""

import sys
import re

version = sys.argv[1] if len(sys.argv) > 1 else ""
# Convert 0.1.1-devci.1 to 0.1.1devci1
# Replace first dash with nothing, then remove all remaining dots after the prerelease identifier
pep440 = re.sub(r"-(\w+)\.(\d+)", r".\1\2", version)
print(pep440)
