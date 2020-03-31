"""Compute ablation points given a total number of documents and a desired number of steps."""

import sys
from decimal import ROUND_HALF_UP, Decimal


def main() -> None:
    orig = Decimal(sys.argv[1])
    n_steps = Decimal(sys.argv[2])
    if orig < n_steps:
        raise ValueError("Number of steps must be smaller than amount to be divided")

    step = orig / n_steps
    total = step
    steps = []

    while total <= orig:
        steps.append(int(total.to_integral_value(ROUND_HALF_UP)))
        total += step

    print(steps)
    if steps[-1] != orig:
        raise ValueError("Final size did not match original")


if __name__ == "__main__":
    main()
