#! /usr/bin/env python

import argparse
import csv
from collections import defaultdict
from typing import DefaultDict

from nerpy import (
    SUPPORTED_ENCODINGS,
    EntityType,
    MentionType,
    get_mention_encoder,
    load_json,
    load_pickled_documents,
    pickle_documents,
    score_prf,
)
from nerpy.annotators.crfsuite import CRFSuiteAnnotator
from nerpy.features import SentenceFeatureExtractor
from nerpy.scoring import ScoringCounts, TokenCounter

NameCounter = DefaultDict[str, DefaultDict[EntityType, int]]


def test(
    model_path: str,
    test_path: str,
    test_pred_path: str,
    feature_params_path: str,
    mention_encoding_name: str,
    output_file: str,
    system_counts_file: str,
    gold_counts_file: str,
) -> None:
    feature_params = load_json(feature_params_path)

    mention_encoder = get_mention_encoder(mention_encoding_name)

    # TODO: Only supports CRFSuite backend right now
    annotator = CRFSuiteAnnotator.from_model(
        MentionType("name"),
        SentenceFeatureExtractor(feature_params),
        mention_encoder(),
        model_path,
    )

    test_docs = load_pickled_documents(test_path)

    pred_docs = [
        annotator.add_mentions(test_doc.copy_without_mentions()) for test_doc in test_docs
    ]
    pickle_documents(pred_docs, test_pred_path)

    res = score_prf(test_docs, pred_docs)
    res.print()

    if (
        output_file is not None
        or system_counts_file is not None
        or gold_counts_file is not None
    ):
        print()
        print("***** Scoring Counts *****")

        scoring_counts: TokenCounter = ScoringCounts().count(pred_docs, test_docs)

        # Initialize counters
        system_counts: NameCounter = defaultdict(lambda: defaultdict(int))
        gold_counts: NameCounter = defaultdict(lambda: defaultdict(int))

        # Calculate system and gold counts
        for entity_type in scoring_counts:
            for token in scoring_counts[entity_type]:
                tp_count = scoring_counts[entity_type][token].true_positives
                fp_count = scoring_counts[entity_type][token].false_positives
                fn_count = scoring_counts[entity_type][token].false_negatives

                system_counts[token][entity_type] = tp_count + fp_count
                gold_counts[token][entity_type] = tp_count + fn_count

    if output_file is not None:
        with open(output_file, "w") as count_file:
            writer = csv.writer(count_file, delimiter=",")
            header = ["token", "entitytype", "tp", "fp", "fn"]
            writer.writerow(header)

            for entity_type in scoring_counts:
                for token in scoring_counts[entity_type]:
                    tp_count = scoring_counts[entity_type][token].true_positives
                    fp_count = scoring_counts[entity_type][token].false_positives
                    fn_count = scoring_counts[entity_type][token].false_negatives
                    line = [
                        str(column)
                        for column in [token, entity_type, tp_count, fp_count, fn_count]
                    ]
                    writer.writerow(line)
            print("Wrote scoring counts to " + output_file)

    if system_counts_file is not None:
        # Get all entity types
        all_entity_types = sorted(list(scoring_counts.keys()))

        with open(system_counts_file, "w") as system_output:
            writer = csv.writer(system_output, delimiter=",")
            header = ["token"] + list(map(str, all_entity_types)) + ["total"]
            writer.writerow(header)

            for token in system_counts:
                token_entity_counts = [
                    system_counts[token][entity_type] for entity_type in all_entity_types
                ]

                line = (
                    [token]
                    + list(map(str, token_entity_counts))
                    + [str(sum(token_entity_counts))]
                )
                writer.writerow(line)
            print("Wrote system counts to " + system_counts_file)

    if gold_counts_file is not None:
        # Get all entity types
        all_entity_types = sorted(list(scoring_counts.keys()))

        with open(gold_counts_file, "w") as gold_output:
            writer = csv.writer(gold_output, delimiter=",")
            header = ["token"] + list(map(str, all_entity_types)) + ["total"]
            writer.writerow(header)

            for token in gold_counts:
                token_entity_counts = [
                    gold_counts[token][entity_type] for entity_type in all_entity_types
                ]
                line = (
                    [token]
                    + list(map(str, token_entity_counts))
                    + [str(sum(token_entity_counts))]
                )
                writer.writerow(line)
            print("Wrote gold counts to " + gold_counts_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test CoNLL")
    parser.add_argument("model", help="Path to model output")
    parser.add_argument("test", help="Path to CoNLL test file")
    parser.add_argument("feature_params", help="Path to feature parameters")
    parser.add_argument(
        "mention_encoding", help="mention encoding", choices=SUPPORTED_ENCODINGS
    )
    parser.add_argument("test_pred", help="Path to CoNLL system predictions")
    parser.add_argument("-o", "--output_file", help="Path to scoring counts output")
    parser.add_argument("-s", "--system_counts_file", help="Path to system counts output")
    parser.add_argument("-g", "--gold_counts_file", help="Path to gold counts output")
    args = parser.parse_args()

    test(
        args.model,
        args.test,
        args.test_pred,
        args.feature_params,
        args.mention_encoding,
        args.output_file,
        args.system_counts_file,
        args.gold_counts_file,
    )


if __name__ == "__main__":
    main()
