#! /usr/bin/env python

import argparse
import os

from nerpy import OntoNotesIngester, pickle_documents

_FORMAT_SUFFIX = ".name"


def ingest_ontonotes(input_path: str, output_path: str) -> None:
    ingester = OntoNotesIngester()
    filename = os.path.basename(input_path)
    docid = (
        filename
        if not filename.endswith(_FORMAT_SUFFIX)
        else filename[: -len(_FORMAT_SUFFIX)]
    )
    with open(input_path, encoding="utf8") as input_file:
        doc = ingester.ingest(input_file, docid)
    pickle_documents([doc], output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CoNLL")
    parser.add_argument("input", help="input OntoNotes .name format file")
    parser.add_argument("output", help="output pickle file")
    args = parser.parse_args()

    ingest_ontonotes(args.input, args.output)


if __name__ == "__main__":
    main()
