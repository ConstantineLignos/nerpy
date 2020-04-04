#! /usr/bin/env python

import argparse
import os

from nerpy import SUPPORTED_ENCODINGS, CoNLLIngester, get_mention_encoder, score_prf


def score_conll(
    reference_path: str, prediction_path: str, encoding_name: str, ignore_comments: bool
) -> None:
    encoder = get_mention_encoder(encoding_name)

    with open(reference_path, encoding="utf8") as reference_file:
        reference_docs = CoNLLIngester(encoder(), ignore_comments=ignore_comments).ingest(
            reference_file, os.path.basename(reference_path)
        )

    with open(prediction_path, encoding="utf8") as prediction_file:
        pred_docs = CoNLLIngester(encoder(), ignore_comments=ignore_comments).ingest(
            prediction_file, os.path.basename(prediction_path)
        )

    res = score_prf(reference_docs, pred_docs)
    print(res)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CoNLL")
    parser.add_argument(
        "reference_file", help="reference (correct results) CoNLL format file"
    )
    parser.add_argument(
        "prediction_file", help="predicted (system output) CoNLL format file"
    )
    parser.add_argument(
        "mention_encoding", help="mention encoding of files", choices=SUPPORTED_ENCODINGS
    )
    parser.add_argument(
        "--ignore-comments", action="store_true", help="ignore comment lines"
    )
    args = parser.parse_args()

    score_conll(
        args.reference_file,
        args.prediction_file,
        args.mention_encoding,
        args.ignore_comments,
    )


if __name__ == "__main__":
    main()
