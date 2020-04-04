import os
from tempfile import TemporaryDirectory

import pytest

from nerpy import DocumentBuilder, EntityType, Mention, MentionType, Sentence, Token
from nerpy.io import (
    load_pickled_document,
    load_pickled_documents,
    pickle_document,
    pickle_documents,
)

NAME = MentionType("name")
DESC = MentionType("desc")
PER = EntityType("PER")
ORG_COM = EntityType(("ORG", "COM"))
MISC = EntityType(("MISC",))


def test_full_document() -> None:
    builder = DocumentBuilder("test")
    assert len(builder) == 0
    assert not bool(builder)

    # Add tokens and sentence
    t1 = Token("foo", 0)
    t2 = Token("bar", 1)
    s1 = builder.create_sentence([t1, t2])
    assert len(builder) == 1
    assert bool(builder)
    # Can't add another sentence with the same index
    with pytest.raises(ValueError):
        builder.add_sentence(s1)
    with pytest.raises(ValueError):
        builder.add_sentences([s1])

    # Add mentions
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t2], NAME, ORG_COM)
    m3 = Mention.create(s1, [t1, t2], DESC, MISC)
    assert not builder.contains_mention(m1)
    builder.add_mention(m1)
    assert builder.contains_mention(m1)

    builder.add_mentions([m2, m3])
    assert builder.contains_mention(m2)
    assert builder.contains_mention(m3)

    # Cannot add a duplicate mention
    with pytest.raises(ValueError):
        builder.add_mention(m1)
    orig_doc = builder.build()

    # Serialize and load the doc
    with TemporaryDirectory() as tmpdir:
        single_doc_path = os.path.join(tmpdir, "doc.pkl")
        pickle_document(orig_doc, single_doc_path)
        unpickled_doc = load_pickled_document(single_doc_path)

        list_doc_path = os.path.join(tmpdir, "docs.pkl")
        pickle_documents([orig_doc], list_doc_path)
        unpickled_docs = load_pickled_documents(list_doc_path)

    for d in orig_doc, unpickled_doc, unpickled_docs[0]:
        # Check tokens and sentence
        assert len(d) == 1
        assert d[0] == s1
        assert list(d) == [s1]
        assert list(d[0]) == [t1, t2]
        assert len(d[0]) == 2
        assert d[0][0] == t1
        assert d[0][1] == t2
        assert list(t1) == ["f", "o", "o"]
        assert len(t1) == 3
        assert t1[1] == "o"

        # Check mentions
        # m1 sorts first, then m3, then m2
        assert d.mentions == (m1, m3, m2)

        # Sentence with mentions
        assert list(d.sentences_with_mentions()) == [(s1, (m1, m3, m2))]

        # Copy with mentions
        m4 = Mention.create(s1, [t1, t2], NAME, PER)
        assert d.copy_with_mentions([m4]).mentions == (m4,)

        # Copy without mentions
        assert d.copy_without_mentions().mentions == ()

        # Test str
        assert str(d) == "foo bar"


def test_sentence_mentions() -> None:
    builder = DocumentBuilder("test")
    # Sentence 1
    t1 = Token("a", 0)
    t2 = Token("b", 1)
    t3 = Token("c", 2)
    s1 = Sentence([t1, t2, t3], 0)
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t2, t3], NAME, MISC)

    # Sentence 1
    t4 = Token("d", 0)
    s2 = Sentence([t4], 1)

    # Sentence 3
    t5 = Token("e", 0)
    s3 = Sentence([t5], 2)
    m3 = Mention.create(s3, [t5], NAME, PER)

    # Cannot add a mention before its sentence
    with pytest.raises(ValueError):
        builder.add_mentions([m1, m2])
    with pytest.raises(ValueError):
        builder.add_mentions([m3])

    # Add sentences and mentions
    builder.add_sentences([s1, s2, s3])
    # Intentionally provide mentions out of order, since the order should not matter
    builder.add_mentions([m3, m1, m2])
    d = builder.build()

    assert list(d.sentences_with_mentions()) == [
        (s1, (m1, m2)),
        (s2, ()),
        (s3, (m3,)),
    ]
    assert d.mentions_for_sentence(s1) == (m1, m2)
    assert d.mentions_for_sentence(s2) == ()
    assert d.mentions_for_sentence(s3) == (m3,)

    # Test using a sentence that doesn't match the document
    s0 = Sentence.from_tokens([Token("z", 0)], 0)
    with pytest.raises(ValueError):
        d.mentions_for_sentence(s0)

    # Test tokens
    m1_tokens = (t1,)
    m2_tokens = (t2, t3)
    assert len(m1) == 1
    assert len(m2) == 2
    assert m1.tokens(d) == m1_tokens
    assert m1.tokens(s1) == m1_tokens
    assert m2.tokens(d) == m2_tokens
    assert m2.tokens(s1) == m2_tokens
    m1_tokenized_text = " ".join([tok.text for tok in m1_tokens])
    m2_tokenized_text = " ".join([tok.text for tok in m2_tokens])
    assert m1.tokenized_text(d) == m1_tokenized_text
    assert m1.tokenized_text(s1) == m1_tokenized_text
    assert m2.tokenized_text(d) == m2_tokenized_text
    assert m2.tokenized_text(s1) == m2_tokenized_text

    # Test getting tokens for a mention from the wrong sentence
    with pytest.raises(ValueError):
        m2.tokens(s2)
    with pytest.raises(ValueError):
        m2.tokenized_text(s2)


def test_token_properties() -> None:
    # Exercise all token fields
    tok = Token.create(
        "dogs",
        0,
        pos_tag="NNS",
        chunk_tag="B-NP",
        lemmas=("dog",),
        properties={"foo": "bar"},
    )
    assert tok.text == "dogs"
    assert tok.index == 0
    assert tok.pos_tag == "NNS"
    assert tok.chunk_tag == "B-NP"
    assert tok.lemmas == ("dog",)
    assert tok.properties["foo"] == "bar"


def test_entity_type() -> None:
    for type_class in (EntityType, MentionType):
        # Simple type, by string or iterable
        t1 = type_class("foo")
        t2 = type_class(["foo"])
        assert t1[0] == "foo"
        assert len(t1) == 1
        assert t2[0] == "foo"
        assert len(t2) == 1
        assert t1 == t2
        assert str(t1) == "foo"
        with pytest.raises(IndexError):
            t1[1]

        # Multi-level type
        t3 = type_class(["foo", "bar"])
        assert t3[0] == "foo"
        assert t3[1] == "bar"
        assert len(t3) == 2
        assert str(t3) == "foo:bar"

        # Invalid initializations
        with pytest.raises(ValueError):
            type_class("")

        with pytest.raises(ValueError):
            type_class(())

        with pytest.raises(TypeError):
            type_class(7)  # type: ignore


def test_bad_token() -> None:
    with pytest.raises(ValueError):
        Token("", 0)

    with pytest.raises(ValueError):
        Token("foo", -1)


def test_bad_sentence() -> None:
    t0 = Token("foo", 0)
    t1 = Token("bar", 1)
    t2 = Token("baz", 2)
    # Tokens in wrong order
    with pytest.raises(ValueError):
        Sentence.from_tokens([t0, t2, t1], 0)
    # Tokens don't start at one
    with pytest.raises(ValueError):
        Sentence.from_tokens([t1, t2], 0)
    # Bad sentence index
    with pytest.raises(ValueError):
        Sentence.from_tokens([t0, t1], -1)


def test_bad_mention() -> None:
    builder = DocumentBuilder("test")
    t0 = Token("foo", 0)
    t1 = Token("bar", 1)
    t2 = Token("baz", 2)
    s1 = builder.create_sentence([t0, t1, t2])
    # No tokens
    with pytest.raises(ValueError):
        Mention.create(s1, [], NAME, PER)

    # Repeated token
    with pytest.raises(ValueError):
        Mention.create(s1, [t0, t0, t1], NAME, PER)

    # Out of order
    with pytest.raises(ValueError):
        Mention.create(s1, [t0, t2, t1], NAME, PER)

    # Skip
    with pytest.raises(ValueError):
        Mention.create(s1, [t0, t2], NAME, PER)

    # Bad token offsets
    with pytest.raises(ValueError):
        Mention(s1.index, 1, 1, NAME, PER)
