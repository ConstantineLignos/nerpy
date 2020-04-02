#! /usr/bin/env python

import argparse
import random
from pathlib import Path
from typing import IO, Optional, TextIO, Union

from nerpy import (
    SUPPORTED_ENCODINGS,
    MentionAnnotator,
    MentionType,
    ScoringResult,
    get_mention_encoder,
    load_json,
    load_pickled_documents,
    pickle_documents,
    score_prf,
)
from nerpy.annotator import SequenceMentionAnnotator
from nerpy.features import SentenceFeatureExtractor

BACKEND_CRFSUITE = "crfsuite"
BACKEND_SEQUENCEMODELS = "sequencemodels"


def train_test(
    train_path: Union[Path, str],
    model_path: Union[Path, str],
    test_path: Union[Path, str],
    train_params_path: Union[Path, str],
    feature_params_path: Union[Path, str],
    mention_encoding_name: str,
    output_path: Union[Path, str],
    *,
    verbose: bool = False,
    truncate: Optional[int] = None,
    log_file: Optional[TextIO] = None,
    random_seed: Optional[int] = None,
) -> ScoringResult:
    annotator = train(
        feature_params_path,
        mention_encoding_name,
        train_params_path,
        train_path,
        verbose,
        truncate,
        log_file,
        random_seed,
    )
    annotator.to_path(model_path)

    res = test(annotator, log_file, output_path, test_path)

    return res


def train(
    feature_params_path: Union[Path, str],
    mention_encoding_name: str,
    train_params_path: Union[Path, str],
    train_path: Union[Path, str],
    verbose: bool = False,
    truncate: Optional[int] = None,
    log_file: Optional[IO[str]] = None,
    random_seed: Optional[int] = None,
) -> SequenceMentionAnnotator:
    mention_encoder = get_mention_encoder(mention_encoding_name)
    feature_params = load_json(feature_params_path)
    train_config = load_json(train_params_path)
    print("Loading training data", file=log_file)
    train_docs = load_pickled_documents(train_path)
    if random_seed is not None:
        print(f"Shuffling documents with random seed {random_seed}", file=log_file)
        random.seed(random_seed)
        random.shuffle(train_docs)
    if truncate is not None:
        print(f"Truncating training to {truncate} documents", file=log_file)
        train_docs = train_docs[:truncate]
    print(
        f"Training using {mention_encoder.__name__} with configuration:\n"
        f"{train_config}",
        file=log_file,
    )
    print(f"Feature configuration:\n{feature_params}", file=log_file)

    mention_type = MentionType("name")
    encoder_instance = mention_encoder()
    feature_extractor = SentenceFeatureExtractor(feature_params)

    backend = train_config["backend"]
    train_params = train_config["train_params"]
    if backend == BACKEND_CRFSUITE:
        from nerpy.annotators.crfsuite import train_crfsuite

        return train_crfsuite(
            encoder_instance,
            feature_extractor,
            mention_type,
            train_docs,
            train_params,
            verbose=verbose,
        )
    elif backend == BACKEND_SEQUENCEMODELS:
        from nerpy.annotators.seqmodels import train_seqmodels

        return train_seqmodels(
            encoder_instance,
            feature_extractor,
            mention_type,
            train_docs,
            train_params,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unrecognized backend: {backend}")


def test(
    annotator: MentionAnnotator,
    log_file: Optional[TextIO],
    output_path: Union[Path, str],
    test_path: Union[Path, str],
) -> ScoringResult:
    print("Loading test data", file=log_file)
    test_docs = load_pickled_documents(test_path)
    pred_docs = [
        annotator.add_mentions(test_doc.copy_without_mentions()) for test_doc in test_docs
    ]
    print("Scoring", file=log_file)
    res = score_prf(test_docs, pred_docs)
    res.print(file=log_file)
    print(file=log_file)
    print(f"Writing test output to {output_path}", file=log_file)
    pickle_documents(pred_docs, output_path)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CoNLL")
    parser.add_argument("train", help="Path to train pickle file")
    parser.add_argument("model", help="Path to model output")
    parser.add_argument("test", help="Path to test pickle file")
    parser.add_argument("train_params", help="Path to training parameters")
    parser.add_argument("feature_params", help="Path to feature parameters")
    parser.add_argument(
        "mention_encoding", help="mention encoding", choices=SUPPORTED_ENCODINGS
    )
    parser.add_argument("output", help="path for predicted test data output")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-t", "--truncate", type=int, help="number of training documents to keep"
    )
    parser.add_argument(
        "-s", "--seed", type=int, help="random seed to use to shuffle data"
    )
    args = parser.parse_args()

    train_test(
        args.train,
        args.model,
        args.test,
        args.train_params,
        args.feature_params,
        args.mention_encoding,
        args.output,
        verbose=args.verbose,
        truncate=args.truncate,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
