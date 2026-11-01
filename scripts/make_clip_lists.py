#!/usr/bin/env python3
"""Write clips_{train,val,test}.json from a dataset split.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("datasets/motions/POL-WALK-EGO"))
    args = p.parse_args()
    split = json.loads((args.data / "split.json").read_text())
    for key in ("train", "val", "test"):
        (args.data / f"clips_{key}.json").write_text(json.dumps(split[key], indent=2))
        print(key, len(split[key]))


if __name__ == "__main__":
    main()
