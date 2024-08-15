from infra.runner import Sweep
from infra.trainers import train_mnist_classifier
import argparse
from enum import Enum


class ExpType(Enum):
    MNIST = "mnist"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_type", type=ExpType, choices=list(ExpType))
    parser.add_argument("exp_name", type=str)
    parser.add_argument("-mw", "--max_workers", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.exp_type == ExpType.MNIST:
        trainer = train_mnist_classifier
        exp_dir = "./experiments/mnist"
    else:
        raise ValueError("Invalid exp type!")

    sweep = Sweep(
        trainer,
        exp_dir,
        args.exp_name,
        max_workers=args.max_workers,
        verbose=args.verbose,
    )
    sweep.start()


if __name__ == "__main__":
    main()
