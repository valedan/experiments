import os
from pathlib import Path
from infra.sweep import Sweep
import argparse

base_sweep_dir = "./sweeps"

def find_sweep_file(sweep_name):
    name_parts: list = sweep_name.strip().split('/')
    sweep_id = name_parts.pop()
    sweep_dir = Path(base_sweep_dir) / '/'.join(name_parts)
    for filename in os.listdir(sweep_dir):
        if filename.startswith(sweep_id) and filename.endswith(".yml"):
            return sweep_dir / filename
    raise FileNotFoundError("Sweep file not found!")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_name", type=str)
    args = parser.parse_args()
    sweep_file = find_sweep_file(args.sweep_name)
    sweep = Sweep(sweep_file)
    sweep.start()


if __name__ == "__main__":
    main()
