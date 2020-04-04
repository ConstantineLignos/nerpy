#! /usr/bin/env python

import argparse

from nerpy import CoNLLIngester, get_mention_encoder


def convert_conll(
    input_path: str,
    input_encoding: str,
    output_path: str,
    output_encoding: str,
    ignore_comments: bool,
) -> None:

    with open(input_path, encoding="utf8") as input_file:
        input_mention_encoding = get_mention_encoder(input_encoding)
        input_docs = CoNLLIngester(
            input_mention_encoding(), ignore_comments=ignore_comments
        ).ingest(input_file, "input")

    output_mention_encoding = get_mention_encoder(output_encoding)()
    # TODO: This should use the same code as write_conll.py
    with open(output_path, "w", encoding="utf8") as output_file:
        for input_doc in input_docs:
            output_file.write("-DOCSTART- -X- -X- O\n\n")
            for sentence, mentions in input_doc.sentences_with_mentions():
                tokens = sentence.tokens
                labels = output_mention_encoding.encode_mentions(sentence, mentions)
                for token, label in zip(tokens, labels):
                    line = " ".join([token.text, token.pos_tag, "None", label])
                    output_file.write(line + "\n")
                output_file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CoNLL")
    parser.add_argument("input_path", help="Path to input file")
    parser.add_argument("input_encoding", help="Input mention encoding")
    parser.add_argument("output_path", help="Path to output file")
    parser.add_argument("output_encoding", help="Output mention encoding")
    parser.add_argument(
        "--ignore-comments", action="store_true", help="ignore comment lines"
    )
    args = parser.parse_args()

    convert_conll(
        args.input_path,
        args.input_encoding,
        args.output_path,
        args.output_encoding,
        args.ignore_comments,
    )


if __name__ == "__main__":
    main()
