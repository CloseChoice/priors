#!/usr/bin/env python3
"""Get the next dev version number from TestPyPI."""

import sys
import json
import re
import urllib.request

if len(sys.argv) < 2:
    print("0")
    sys.exit(0)

base_version = sys.argv[1]
package_name = "priors"
api_url = f"https://test.pypi.org/pypi/{package_name}/json"

try:
    with urllib.request.urlopen(api_url) as response:
        data = json.load(response)
        versions = data.get("releases", {}).keys()

        # Filter for dev versions matching our base version
        pattern = re.escape(base_version) + r"\.dev(\d+)"

        dev_numbers = []
        for v in versions:
            match = re.match(pattern, v)
            if match:
                dev_numbers.append(int(match.group(1)))

        print(max(dev_numbers) if dev_numbers else 0)
except Exception:
    print(0)
