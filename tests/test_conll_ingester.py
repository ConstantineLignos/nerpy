# TODO: Rename to test_conll.py

import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import pytest

from nerpy import (
    BIO,
    IOB,
    CoNLLIngester,
    DocumentBuilder,
    EntityType,
    Mention,
    MentionEncoder,
    MentionType,
    Token,
)
from nerpy.ingest.conll import read_conll, write_conll
from nerpy.io import PathType

TEST_DATA_DIR = os.path.join("tests", "test_data")

# TODO: Build all paths using Path


def test_label_conversion():
    # Read IOB1 tags
    ingest = CoNLLIngester(IOB())
    with open("tests/test_data/en_iob.txt", encoding="utf-8") as input_file:
        tags_iob1 = [
            [token.ne_tag for token in sentence]
            for sentence in ingest._parse_file(input_file)
        ]

    # Read IOB2 tags
    ingest = CoNLLIngester(BIO())
    with open("tests/test_data/en_bio.txt", encoding="utf-8") as input_file:
        tags_iob2 = [
            [token.ne_tag for token in sentence]
            for sentence in ingest._parse_file(input_file)
        ]

    # Convert IOB1 to IOB2 and check if its same as original IOB2
    for i, tags in enumerate(tags_iob1):
        res = iob1_iob2(tags)
        assert res == tags_iob2[i]

    # Convert IOB2to IOB1 and check if its same as original IOB1
    for i, tags in enumerate(tags_iob2):
        res = iob2_iob1(tags)
        assert res == tags_iob1[i]


def test_ingest():
    ingest = CoNLLIngester(IOB())
    iob1_docs = ingest.ingest(open("tests/test_data/en_iob.txt"), "test_iob")

    ingest = CoNLLIngester(BIO())
    iob2_docs = ingest.ingest(open("tests/test_data/en_bio.txt"), "test_bio")

    for doc1, doc2 in zip(iob1_docs, iob2_docs):
        assert doc1.mentions == doc2.mentions


def test_bad_line():
    # DOCSTART is part of sentence, should only be between sentences
    text = io.StringIO(
        """-DOCSTART- -X- -X- O

EU NNP B-NP B-ORG
rejects VBZ B-VP O
-DOCSTART- -X- -X- O
"""
    )
    ingest = CoNLLIngester(IOB())
    with pytest.raises(ValueError):
        ingest.ingest(text, "test")


def test_ignore_comments():
    text = """# Comment here
EU NNP B-NP B-ORG
rejects VBZ B-VP O

# Comment here
"""

    # Default behavior fails on comments
    ingest = CoNLLIngester(BIO())
    with pytest.raises(ValueError):
        ingest.ingest(io.StringIO(text), "test")

    # No problem if flag specified
    ingest = CoNLLIngester(BIO(), ignore_comments=True)
    docs = ingest.ingest(io.StringIO(text), "test")
    assert len(docs) == 1
    print(docs[0])
    assert len(docs[0]) == 1


def test_parse_line():
    # English examples
    eng_docstart = "-DOCSTART- -X- -X- O"
    eng_line = "EU NNP B-NP B-ORG"

    # Dutch examples
    ned_docstart = "-DOCSTART- -DOCSTART- O"
    ned_line = "De Art O"

    # Spanish examples (no document boundaries in Spanish)
    spa_line = "El DA O"

    # German examples
    deu_docstart = "-DOCSTART- -X- -X- -X- O"
    deu_line1 = "Schwierigkeiten Schwierigkeit NN I-NC O"
    deu_line2 = "Einwanderungsfragen Einwanderungsfragen|Einwanderungsfragen NN I-NC O"

    tok = CoNLLIngester._CoNLLToken.from_line(eng_docstart, 1)
    assert tok.is_docstart

    tok = CoNLLIngester._CoNLLToken.from_line(eng_line, 1)
    assert tok.text == "EU"
    assert tok.lemmas is None
    assert tok.pos_tag == "NNP"
    assert tok.chunk_tag == "B-NP"
    assert tok.ne_tag == "B-ORG"

    tok = CoNLLIngester._CoNLLToken.from_line(ned_docstart, 1)
    assert tok.is_docstart

    tok = CoNLLIngester._CoNLLToken.from_line(ned_line, 1)
    assert tok.text == "De"
    assert tok.lemmas is None
    assert tok.pos_tag == "Art"
    assert tok.chunk_tag is None
    assert tok.ne_tag == "O"

    tok = CoNLLIngester._CoNLLToken.from_line(spa_line, 1)
    assert tok.text == "El"
    assert tok.lemmas is None
    assert tok.pos_tag == "DA"
    assert tok.chunk_tag is None
    assert tok.ne_tag == "O"

    tok = CoNLLIngester._CoNLLToken.from_line(deu_docstart, 1)
    assert tok.is_docstart

    tok = CoNLLIngester._CoNLLToken.from_line(deu_line1, 1)
    assert tok.text == "Schwierigkeiten"
    assert tok.lemmas == ("Schwierigkeit",)
    assert tok.pos_tag == "NN"
    assert tok.chunk_tag == "I-NC"
    assert tok.ne_tag == "O"

    tok = CoNLLIngester._CoNLLToken.from_line(deu_line2, 1)
    assert tok.text == "Einwanderungsfragen"
    assert tok.lemmas == ("Einwanderungsfragen", "Einwanderungsfragen")
    assert tok.pos_tag == "NN"
    assert tok.chunk_tag == "I-NC"
    assert tok.ne_tag == "O"


def iob1_iob2(tags: list) -> List[str]:
    """
    Tags in IOB1 format are converted to IOB2.
    """
    res = tags[:]

    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        split = tag.split("-")
        if split[0] == "B":
            continue
        elif i == 0 or tags[i - 1] == "O":
            res[i] = "B" + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:
            res[i] = "B" + tag[1:]
    return res


def iob2_iob1(tags: list) -> List[str]:
    """
    Tags in IOB2 format are converted to IOB1.
    """
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        if i == 0 or tags[i - 1] == "O":
            tags[i] = "I" + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:
            tags[i] = "I" + tag[1:]
    return tags


# This just tests the test utilities
def test_iob_conversion():
    iob1 = ["I-PER", "I-PER", "I-LOC", "O", "I-LOC", "I-LOC", "I-LOC", "O"]
    iob2 = ["B-PER", "I-PER", "B-LOC", "O", "B-LOC", "I-LOC", "I-LOC", "O"]
    new_iob2 = iob1_iob2(iob1)
    assert iob2 == new_iob2

    iob1 = ["I-PER", "I-PER", "I-LOC", "O", "I-LOC", "I-LOC", "I-LOC", "O"]
    iob2 = ["B-PER", "I-PER", "B-LOC", "O", "B-LOC", "I-LOC", "I-LOC", "O"]
    new_iob1 = iob2_iob1(iob2)
    assert iob1 == new_iob1


def test_write_conll() -> None:
    eng_bio_path = Path(TEST_DATA_DIR, "en_bio.txt")
    eng_iob_path = Path(TEST_DATA_DIR, "en_iob.txt")

    # Test round-tripping
    with TemporaryDirectory() as tmpdir:
        input_paths = (eng_bio_path, eng_iob_path)
        output_paths = [Path(tmpdir, path.name) for path in input_paths]
        mention_encoders = (BIO(), IOB())
        for input_path, output_path, mention_encoder in zip(
            input_paths, output_paths, mention_encoders
        ):
            _round_trip_conll_file(input_path, output_path, mention_encoder)


def _round_trip_conll_file(
    input_path: PathType, output_path: PathType, mention_encoder: MentionEncoder
) -> None:
    docs1 = read_conll(input_path, mention_encoder)
    write_conll(docs1, output_path, mention_encoder)
    docs2 = read_conll(output_path, mention_encoder)
    assert docs2 == docs1


def test_nested_names() -> None:
    encoder = BIO()
    name = MentionType("name")
    per = EntityType("PER")
    builder = DocumentBuilder("test")
    t1 = Token("token1", 0)
    t2 = Token("token2", 1)
    s1 = builder.create_sentence([t1, t2])
    m1 = Mention.create(s1, [t1, t2], name, per)
    m2 = Mention.create(s1, [t2], name, per)
    builder.add_mentions([m1, m2])
    doc = builder.build()
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir, "tmp.conll")
        with pytest.raises(ValueError):
            write_conll([doc], output_path, encoder)


def test_ger() -> None:
    ger_path = Path(TEST_DATA_DIR, "deu_iob.txt")
    docs = read_conll(ger_path, IOB())
    # One doc
    assert len(docs) == 1
    doc = docs[0]

    # One sentence
    assert len(doc) == 1
    # One mention in total
    assert len(doc.mentions) == 1

    sent, mentions = next(iter(doc.sentences_with_mentions()))
    # 19 tokens
    assert len(sent) == 19
    # One mention in sentence
    assert len(mentions) == 1
    # Check lemma on token 6
    assert sent[6].lemmas == ("Einwanderungsfrage", "Einwanderungsfragen")

    with TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir, ger_path.name)
        _round_trip_conll_file(ger_path, out_path, IOB())
