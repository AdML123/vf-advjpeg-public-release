from __future__ import annotations

import argparse
from dataclasses import asdict
import json

import _bootstrap  # noqa: F401

from vf_advjpeg.code_ocean import run_code_ocean_reproduction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/code_ocean_capsule.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    outputs = run_code_ocean_reproduction(
        config_path=args.config,
        overwrite=args.overwrite,
        max_batches=args.max_batches,
        skip_compare=args.skip_compare,
    )
    print(json.dumps({key: str(value) for key, value in asdict(outputs).items()}, indent=2))


if __name__ == "__main__":
    main()
