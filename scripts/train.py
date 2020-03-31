#! /usr/bin/env python

import argparse

from nerpy import (
    SUPPORTED_ENCODINGS,
    MentionType,
    get_mention_encoder,
    load_json,
    load_pickled_documents,
)
from nerpy.features import SentenceFeatureExtractor

BACKEND_CRFSUITE = "crfsuite"
BACKEND_SEQUENCEMODELS = "sequencemodels"


def train(
    train_path: str,
    model_path: str,
    train_params_path: str,
    feature_params_path: str,
    mention_encoding_name: str,
    *,
    verbose: bool = False,
) -> None:
    mention_encoder = get_mention_encoder(mention_encoding_name)
    feature_params = load_json(feature_params_path)
    train_config = load_json(train_params_path)
    train_docs = load_pickled_documents(train_path)

    mention_type = MentionType("name")
    encoder_instance = mention_encoder()
    feature_extractor = SentenceFeatureExtractor(feature_params)

    backend = train_config["backend"]
    train_params = train_config["train_params"]
    if backend == BACKEND_CRFSUITE:
        from nerpy.annotators.crfsuite import train_crfsuite

        train_crfsuite(
            encoder_instance,
            feature_extractor,
            mention_type,
            model_path,
            train_docs,
            train_params,
            verbose=verbose,
        )
    elif backend == BACKEND_SEQUENCEMODELS:
        from nerpy.annotators.seqmodels import train_seqmodels

        train_seqmodels(
            encoder_instance,
            feature_extractor,
            mention_type,
            model_path,
            train_docs,
            train_params,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unrecognized backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CoNLL")
    parser.add_argument("train", help="Path to pickled training data file")
    parser.add_argument("model", help="Path to write model to")
    parser.add_argument("train_params", help="Path to training parameters")
    parser.add_argument("feature_params", help="Path to feature parameters")
    parser.add_argument(
        "mention_encoder", help="mention encoder", choices=SUPPORTED_ENCODINGS
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    train(
        args.train,
        args.model,
        args.train_params,
        args.feature_params,
        args.mention_encoder,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
