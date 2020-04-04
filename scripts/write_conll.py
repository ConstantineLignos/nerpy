#! /usr/bin/env python

import argparse

from nerpy import get_mention_encoder
from nerpy.ingest.conll import write_conll
from nerpy.io import load_pickled_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CoNLL")
    parser.add_argument("input_path", help="Path to input file")
    parser.add_argument("output_path", help="Path to output file")
    parser.add_argument("--encoding", default="BIO", help="Output mention encoding")
    parser.add_argument(
        "--lang",
        help="Language code if needed to to specify unusual formats. "
        "Currently only has any effect for language 'ned'.",
    )
    parser.add_argument(
        "--no-docstart", action="store_true", help="Do not output -DOCSTART- lines"
    )
    args = parser.parse_args()

    mention_encoder = get_mention_encoder(args.encoding)()
    docs = load_pickled_documents(args.input_path)
    write_conll(docs, args.output_path, mention_encoder, args.lang)


if __name__ == "__main__":
    main()
