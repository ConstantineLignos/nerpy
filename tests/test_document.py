import pytest

from nerpy import DocumentBuilder, EntityType, Mention, MentionType, Sentence, Token

NAME = MentionType("name")
DESC = MentionType("desc")
PER = EntityType("PER")
ORG_COM = EntityType(("ORG", "COM"))
MISC = EntityType(("MISC",))


def test_full_document() -> None:
    builder = DocumentBuilder("test")
    t1 = Token("foo", 0)
    t2 = Token("bar", 1)
    s1 = builder.create_sentence([t1, t2])
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t2], NAME, ORG_COM)
    m3 = Mention.create(s1, [t1, t2], DESC, MISC)
    builder.add_mention(m1)
    builder.add_mentions([m2, m3])
    d = builder.build()

    # Check tokens and sentence
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
    assert len(d.mentions) == 3
    assert d.mentions[0].tokens(d) == (t1,)
    assert d.mentions[0].mention_type == NAME
    assert d.mentions[0].entity_type == PER
    assert len(d.mentions[0]) == 1
    assert d.mentions[1].tokens(d) == (t2,)
    assert d.mentions[1].mention_type == NAME
    assert d.mentions[1].entity_type == ORG_COM
    assert len(d.mentions[1]) == 1
    assert d.mentions[2].tokens(d) == (t1, t2)
    assert d.mentions[2].mention_type == DESC
    assert d.mentions[2].entity_type == MISC
    assert len(d.mentions[2]) == 2

    # Sentence with mentions
    assert list(d.sentences_with_mentions()) == [(s1, (m1, m2, m3))]

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
    s1 = builder.create_sentence([t1, t2])
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t2], NAME, MISC)

    # Sentence 1
    t3 = Token("c", 0)
    s2 = builder.create_sentence([t3])

    # Sentence 3
    t4 = Token("d", 0)
    s3 = builder.create_sentence([t4])
    m3 = Mention.create(s3, [t4], NAME, PER)

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
    # Simple type, by string or iterable
    et1 = EntityType("foo")
    et2 = EntityType(["foo"])
    assert et1[0] == "foo"
    assert len(et1) == 1
    assert et2[0] == "foo"
    assert len(et2) == 1
    assert et1 == et2
    assert str(et1) == "foo"
    with pytest.raises(IndexError):
        et1[1]

    # Multi-level type
    et3 = EntityType(["foo", "bar"])
    assert et3[0] == "foo"
    assert et3[1] == "bar"
    assert len(et3) == 2
    assert str(et3) == "foo.bar"


def test_bad_token() -> None:
    with pytest.raises(ValueError):
        Token("", 0)

    with pytest.raises(ValueError):
        Token("foo", -1)


def test_bad_mention_type() -> None:
    with pytest.raises(ValueError):
        MentionType("")

    with pytest.raises(TypeError):
        MentionType(7)  # type: ignore


def test_bad_entity_type() -> None:
    with pytest.raises(ValueError):
        EntityType("")

    with pytest.raises(ValueError):
        EntityType(())

    with pytest.raises(TypeError):
        EntityType(7)  # type: ignore


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
