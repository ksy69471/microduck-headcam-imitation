#!/usr/bin/env bash
# Clone official Microduck repos into ./readonly (idempotent).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/readonly"
cd "$ROOT/readonly"

if [[ ! -d microduck_rl/.git ]]; then
  git clone --depth 1 https://github.com/pollen-robotics/microduck_rl microduck_rl
else
  echo "skip: readonly/microduck_rl already present"
fi

if [[ ! -d microduck/.git ]]; then
  git clone --depth 1 https://github.com/pollen-robotics/microduck microduck
else
  echo "skip: readonly/microduck already present"
fi

echo "OK: third-party clones under $ROOT/readonly"
