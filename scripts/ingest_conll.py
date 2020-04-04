#! /usr/bin/env python

import argparse
import os
import time

from nerpy import (
    SUPPORTED_ENCODINGS,
    CoNLLIngester,
    get_mention_encoder,
    pickle_documents,
)


def ingest_conll(
    input_path: str, output_path: str, encoding_name: str, ignore_comments: bool
) -> None:
    encoder = get_mention_encoder(encoding_name)

    print(f"Loading data from {input_path} using mention encoding {encoder.__name__}")
    start_time = time.perf_counter()
    with open(input_path, encoding="utf8") as train_file:
        input_docs = CoNLLIngester(encoder(), ignore_comments=ignore_comments).ingest(
            train_file, os.path.basename(input_path)
        )
    print(
        f"Loaded {len(input_docs)} documents in {time.perf_counter() - start_time} seconds"
    )

    start_time = time.perf_counter()
    pickle_documents(input_docs, output_path)
    print(f"Wrote output to {output_path} in {time.perf_counter() - start_time} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CoNLL")
    parser.add_argument("input", help="input CoNLL format file")
    parser.add_argument(
        "mention_encoding",
        help="mention encoding of input file",
        choices=SUPPORTED_ENCODINGS,
    )
    parser.add_argument("output", help="output pickle file")
    parser.add_argument(
        "--ignore-comments", action="store_true", help="ignore comment lines"
    )
    args = parser.parse_args()

    ingest_conll(args.input, args.output, args.mention_encoding, args.ignore_comments)


if __name__ == "__main__":
    main()
