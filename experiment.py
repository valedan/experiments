import os
from pathlib import Path
from infra.runner import Sweep
import argparse

base_exp_dir = "./experiments"

def find_exp_file(exp_name):
    name_parts: list = exp_name.strip().split('/')
    exp_id = name_parts.pop()
    exp_dir = Path(base_exp_dir) / '/'.join(name_parts)
    for filename in os.listdir(exp_dir):
        if filename.startswith(exp_id) and filename.endswith(".yml"):
            return exp_dir / filename
    return None



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("-w", "--max_workers", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_false")
    args = parser.parse_args()
    exp_file = find_exp_file(args.exp_name)
    if not exp_file:
        raise ValueError("Exp file not found")
    sweep = Sweep(
        exp_file,
        max_workers=args.max_workers,
        verbose=args.verbose,
    )
    sweep.start()


if __name__ == "__main__":
    main()
