#! /usr/bin/env python
"""
Run NER experiments.
"""

import argparse
import csv
import multiprocessing
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

from attr import attrs

from nerpy import load_json
from nerpy.scoring import ScoringResult
from scripts.train_test import train_test


@attrs(auto_attribs=True)
class ExperimentConfiguration:
    name: str
    train_path: str
    test_path: str
    training_params_path: str
    feature_params_path: str
    mention_encoding_name: str
    ablation_size: int
    random_seed: Optional[int]
    output_dir: str


def run_expts(
    name_prefix: str,
    train_path: str,
    test_path: str,
    training_params_path: str,
    feature_params_path: str,
    mention_encodings_path: str,
    ablation_path: str,
    random_seed_path: str,
    workers: int,
    output_base: str,
) -> None:
    # Load experiment conditions
    mention_encodings = load_json(mention_encodings_path)
    ablation_sizes = load_json(ablation_path)
    assert isinstance(ablation_sizes, list)
    random_seeds = load_json(random_seed_path)

    # Create configurations
    configs = []
    for random_seed in random_seeds:
        # Hack to allow for no shuffle: replace random seed of -1 with None
        random_seed = None if random_seed == -1 else random_seed

        # Reverse ablation sizes so longest jobs start first
        for ablation_size in reversed(ablation_sizes):
            for mention_encoding in mention_encodings:
                # Name
                config_name = "_".join(
                    (
                        name_prefix,
                        f"rand-{random_seed}",
                        f"ablation-{ablation_size}",
                        mention_encoding,
                    )
                )
                # Output path
                output_dir = os.path.join(output_base, config_name)
                config = ExperimentConfiguration(
                    config_name,
                    train_path,
                    test_path,
                    training_params_path,
                    feature_params_path,
                    mention_encoding,
                    int(ablation_size),
                    random_seed,
                    output_dir,
                )
                configs.append(config)

    print(f"Generated {len(configs)} configurations")

    # To conserve memory, only allow one task per child
    pool = multiprocessing.Pool(workers, maxtasksperchild=1)
    # Chunksize of 1 since each task is long-lived
    results = pool.imap_unordered(run_configuration, configs, chunksize=1)
    pool.close()

    output_dir = Path(output_base)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scores.csv"
    # Line buffered output so results appear quickly
    with open(csv_path, "w", encoding="utf8", buffering=1) as score_csv:
        writer = csv.writer(score_csv)
        writer.writerow(
            [
                "Name",
                "Features",
                "Training",
                "Algorithm",
                "Documents",
                "Random",
                "Encoding",
                "P",
                "R",
                "F1",
            ]
        )

        for config, score in results:
            print("*" * 40)
            print(config.name)
            score.print()
            print()
            writer.writerow(
                [
                    config.name,
                    os.path.basename(config.feature_params_path),
                    os.path.basename(config.training_params_path),
                    config.ablation_size,
                    config.random_seed if config.random_seed is not None else -1,
                    config.mention_encoding_name,
                    score.precision,
                    score.recall,
                    score.fscore,
                ]
            )

    pool.join()


def run_configuration(
    config: ExperimentConfiguration,
) -> Tuple[ExperimentConfiguration, ScoringResult]:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = str(out / "model.crfsuite")  # crfsuite demands a string
    pred_path = out / "test.nerpydoc"
    with (out / "train.log").open("w", encoding="utf8", buffering=1) as log_file, (
        out / "score.pkl"
    ).open("wb") as score_file:
        score = train_test(
            config.train_path,
            model_path,
            config.test_path,
            config.training_params_path,
            config.feature_params_path,
            config.mention_encoding_name,
            pred_path,
            truncate=config.ablation_size,
            log_file=log_file,
            random_seed=config.random_seed,
        )
        pickle.dump(score, score_file, protocol=pickle.HIGHEST_PROTOCOL)
        return (config, score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="prefix for configuration names")
    parser.add_argument("train_path", help="path to train pickle file")
    parser.add_argument("test_path", help="path to test pickle file")
    parser.add_argument("train_params", help="path to training parameters")
    parser.add_argument("feature_params", help="path to feature parameters")
    parser.add_argument("mention_encoding_path", help="path to list of mention encodings")
    parser.add_argument("ablation_path", help="path to list of ablation points")
    parser.add_argument("random_seed_path", help="path to list of random seeds")
    parser.add_argument("output_base", help="base output directory")
    parser.add_argument(
        "-n", "--num_workers", type=int, default=1, help="number of workers"
    )
    args = parser.parse_args()

    run_expts(
        args.prefix,
        args.train_path,
        args.test_path,
        args.train_params,
        args.feature_params,
        args.mention_encoding_path,
        args.ablation_path,
        args.random_seed_path,
        args.num_workers,
        args.output_base,
    )


if __name__ == "__main__":
    main()
