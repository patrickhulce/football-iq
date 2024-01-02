#!/bin/bash

set -uo pipefail

read -r -d '' FRAGMENT << EOM
[filter "strip-notebook-output"]
  clean = "jq '.cells |= map(if .\"cell_type\" == \"code\" then .outputs = [] | .execution_count = null else . end | .metadata = {}) | .metadata = {}'"
EOM

TARGET=".git/config"

# Check if the fragment is already in the file.
if grep -Fq "strip-notebook-output" "$TARGET"; then
  echo "Fragment already in $TARGET, skipping install."
else
  echo "Adding fragment to $TARGET"
  echo "$FRAGMENT" >> "$TARGET"
fi
