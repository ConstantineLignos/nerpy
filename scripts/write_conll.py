#! /usr/bin/env python

import argparse
from typing import Optional

from nerpy import get_mention_encoder
from nerpy.ingest.conll import DOCSTART
from nerpy.io import load_pickled_documents


def write_conll(
    input_path: str,
    output_path: str,
    mention_encoding_name: str,
    lang: Optional[str] = None,
) -> None:
    mention_encoder = get_mention_encoder(mention_encoding_name)()
    docs = load_pickled_documents(input_path)

    docstart_fields = [DOCSTART]
    # Figure out how many fields to output by seeing how many of the CoNLL
    # fields are present
    sample_tok = docs[0][0][0]
    has_lemmas = sample_tok.lemmas is not None
    has_pos = sample_tok.pos_tag is not None
    has_chunk = sample_tok.chunk_tag is not None
    # Add as many additional fields as needed by counting the number of Trues
    # The additional fields are -X- normally, but -DOCSTART- for the Dutch data
    for _ in range(sum((has_lemmas, has_pos, has_chunk))):
        docstart_fields.append("-X-" if lang != "ned" else DOCSTART)
    docstart_fields.append("O")
    docstart_line = " ".join(docstart_fields)

    with open(output_path, "w", encoding="utf8") as output_file:
        for doc in docs:
            # Don't output docstart if there's only one document
            if len(docs) > 1:
                print(docstart_line, file=output_file)
                # For Dutch data, no blank line after docstart
                if lang != "ned":
                    print(file=output_file)
            for sentence, mentions in doc.sentences_with_mentions():
                tokens = sentence.tokens
                labels = mention_encoder.encode_mentions(sentence, mentions)
                for token, label in zip(tokens, labels):
                    line_fields = [token.text]
                    if has_lemmas:
                        line_fields.append("|".join(token.lemmas))
                    if has_pos:
                        line_fields.append(token.pos_tag)
                    if has_chunk:
                        line_fields.append(token.chunk_tag)
                    line_fields.append(label)
                    line = " ".join(line_fields)
                    print(line, file=output_file)
                print(file=output_file)


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

    write_conll(args.input_path, args.output_path, args.encoding, args.lang)


if __name__ == "__main__":
    main()
