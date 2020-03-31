#! /usr/bin/env python

import argparse
import csv
from collections import defaultdict
from typing import Any, DefaultDict, Set, Tuple

# TODO: Fix weird types in values of dictionary
# Right now the values can be int, float, or str
NameCounter = DefaultDict[str, DefaultDict[str, Any]]


def compare(
    system_counts_file: str, gold_counts_file: str, delta_counts_file: str
) -> None:
    # Populate counts
    system_counts, system_entity_types = compute_counts(system_counts_file)
    gold_counts, gold_entity_types = compute_counts(gold_counts_file)
    all_entity_types = system_entity_types | gold_entity_types

    # Compute delta counts
    delta_counts = _create_namecounter()
    for token in system_counts:
        for entity_type in system_counts[token]:
            if token not in gold_counts:
                raise ValueError("Token not present in gold_counts csv")

            system_count = system_counts[token][entity_type]
            gold_count = gold_counts[token][entity_type]

            if gold_count == 0:
                delta_count = ""
            else:
                delta_count = (system_count - gold_count) / gold_count

            delta_counts[token][entity_type] = delta_count
            delta_counts[token]["system_total"] = system_counts[token]["total"]
            delta_counts[token]["gold_total"] = gold_counts[token]["total"]

    sorted_entity_types = sorted(all_entity_types)
    with open(delta_counts_file, "w") as delta_output:
        writer = csv.writer(delta_output, delimiter=",")
        header = (
            ["token"]
            + [entity_type + "_delta" for entity_type in sorted_entity_types]
            + ["system_total", "gold_total"]
        )
        writer.writerow(header)

        for token in delta_counts:
            token_entity_counts = [
                delta_counts[token][entity_type] for entity_type in sorted_entity_types
            ]
            line = (
                [token]
                + list(map(str, token_entity_counts))
                + [str(system_counts[token]["total"]), str(gold_counts[token]["total"])]
            )
            writer.writerow(line)
        print("Wrote delta counts to " + delta_counts_file)


def compute_counts(path: str) -> Tuple[NameCounter, Set[str]]:
    counts = _create_namecounter()
    all_entity_types = set()
    with open(path, encoding="utf8") as system_output:
        reader = csv.reader(system_output, delimiter=",")
        header = next(reader)

        for row in reader:
            token = row[0]
            total = row[-1]
            for i in range(1, len(row) - 1):
                entity_type = header[i]
                counts[token][entity_type] = int(row[i])
                all_entity_types.add(entity_type)
            counts[token]["total"] = int(total)

    return counts, all_entity_types


def _create_namecounter() -> NameCounter:
    return defaultdict(lambda: defaultdict(int))


def main() -> None:
    parser = argparse.ArgumentParser(description="Counts Comparison")
    parser.add_argument("system_counts_file", help="Path to system counts output")
    parser.add_argument("gold_counts_file", help="Path to gold counts output")
    parser.add_argument("delta_counts_file", help="Path to compare counts output")
    args = parser.parse_args()

    compare(args.system_counts_file, args.gold_counts_file, args.delta_counts_file)


if __name__ == "__main__":
    main()
