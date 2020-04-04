import pytest

from nerpy import (
    BILOU,
    BIO,
    BIOES,
    BIOU,
    BMES,
    IO,
    IOB,
    DocumentBuilder,
    EntityType,
    Mention,
    MentionType,
    Token,
    get_mention_encoder,
)

NAME = MentionType("name")
PER = EntityType("PER")
LOC = EntityType("LOC")
MISC = EntityType("MISC")
ORG = EntityType("ORG")

builder = DocumentBuilder("test")
t1 = Token("token1", 0)
t2 = Token("token2", 1)
t3 = Token("token3", 2)
t4 = Token("token4", 3)
t5 = Token("token5", 4)
t6 = Token("token6", 5)
t7 = Token("token7", 6)

s1 = builder.create_sentence([t1, t2, t3, t4, t5, t6, t7])


def test_generic_decoder_bad_labels():
    # All share the same decoder, so which one we choose doesn't matter
    encoder = BIO()

    # Test bad labels
    labels1 = ["O", "T", "I-PER", "O", "I-MISC", "O", "I-LOC"]
    with pytest.raises(ValueError):
        encoder.decode_mentions(s1, labels1)
    labels2 = ["O", "I-", "I-PER", "O", "I-MISC", "O", "I-LOC"]
    with pytest.raises(ValueError):
        encoder.decode_mentions(s1, labels2)
    labels2 = ["O", "I", "I-PER", "O", "I-MISC", "O", "I-LOC"]
    with pytest.raises(ValueError):
        encoder.decode_mentions(s1, labels2)

    # Different number of tokens and labels
    labels4 = ["O", "B-PER"]
    with pytest.raises(ValueError):
        encoder.decode_mentions(s1, labels4)


def test_io_encoder():
    encoder = IO()

    labels1 = ["O", "I-PER", "I-PER", "O", "I-MISC", "O", "I-LOC"]
    m1 = Mention.create(s1, [t2, t3], NAME, PER)
    m2 = Mention.create(s1, [t5], NAME, MISC)
    m3 = Mention.create(s1, [t7], NAME, LOC)
    assert encoder.decode_mentions(s1, labels1) == [m1, m2, m3]
    assert encoder.encode_mentions(s1, [m1, m2, m3]) == tuple(labels1)

    labels2 = ["I-PER", "I-PER", "I-PER", "I-MISC", "I-LOC", "O", "I-PER"]
    m1 = Mention.create(s1, [t1, t2, t3], NAME, PER)
    m2 = Mention.create(s1, [t4], NAME, MISC)
    m3 = Mention.create(s1, [t5], NAME, LOC)
    m4 = Mention.create(s1, [t7], NAME, PER)
    assert encoder.decode_mentions(s1, labels2) == [m1, m2, m3, m4]
    assert encoder.encode_mentions(s1, [m1, m2, m3, m4]) == tuple(labels2)

    labels3 = ["O", "O", "O", "O", "O", "O", "O"]
    assert encoder.decode_mentions(s1, labels3) == []
    assert encoder.encode_mentions(s1, []) == tuple(labels3)


def test_biou_encoder():
    encoder = BIOU()

    labels1 = ["O", "B-PER", "I-PER", "O", "U-MISC", "U-LOC", "U-PER"]
    m1 = Mention.create(s1, [t2, t3], NAME, PER)
    m2 = Mention.create(s1, [t5], NAME, MISC)
    m3 = Mention.create(s1, [t6], NAME, LOC)
    m4 = Mention.create(s1, [t7], NAME, PER)
    assert encoder.decode_mentions(s1, labels1) == [m1, m2, m3, m4]
    assert encoder.encode_mentions(s1, [m1, m2, m3, m4]) == tuple(labels1)

    labels2 = ["B-PER", "I-PER", "I-PER", "I-PER", "U-MISC", "B-LOC", "I-LOC"]
    m1 = Mention.create(s1, [t1, t2, t3, t4], NAME, PER)
    m2 = Mention.create(s1, [t5], NAME, MISC)
    m3 = Mention.create(s1, [t6, t7], NAME, LOC)
    assert encoder.decode_mentions(s1, labels2) == [m1, m2, m3]
    assert encoder.encode_mentions(s1, [m1, m2, m3]) == tuple(labels2)

    labels3 = ["O", "O", "O", "O", "O", "O", "O"]
    assert encoder.decode_mentions(s1, labels3) == []
    assert encoder.encode_mentions(s1, []) == tuple(labels3)

    # Test edge cases from bad decoding
    labels4 = ["I-PER", "B-PER", "I-ORG", "B-LOC", "U-LOC", "O", "I-MISC"]
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t2], NAME, PER)
    m3 = Mention.create(s1, [t3], NAME, ORG)
    m4 = Mention.create(s1, [t4], NAME, LOC)
    m5 = Mention.create(s1, [t5], NAME, LOC)
    m6 = Mention.create(s1, [t7], NAME, MISC)
    assert encoder.decode_mentions(s1, labels4) == [m1, m2, m3, m4, m5, m6]

    # Test edge cases from bad decoding
    labels5 = ["B-PER", "I-ORG", "I-ORG", "U-LOC", "U-PER", "B-ORG", "I-MISC"]
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t2, t3], NAME, ORG)
    m3 = Mention.create(s1, [t4], NAME, LOC)
    m4 = Mention.create(s1, [t5], NAME, PER)
    m5 = Mention.create(s1, [t6], NAME, ORG)
    m6 = Mention.create(s1, [t7], NAME, MISC)
    assert encoder.decode_mentions(s1, labels5) == [m1, m2, m3, m4, m5, m6]

    # Test edge cases from bad decoding
    labels6 = ["B-PER", "O", "I-ORG", "O", "I-LOC", "O", "B-MISC"]
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t3], NAME, ORG)
    m3 = Mention.create(s1, [t5], NAME, LOC)
    m4 = Mention.create(s1, [t7], NAME, MISC)
    assert encoder.decode_mentions(s1, labels6) == [m1, m2, m3, m4]


def test_bilou_encoder():
    encoder = BILOU()

    labels1 = ["O", "B-PER", "L-PER", "O", "U-MISC", "U-LOC", "U-PER"]
    m1 = Mention.create(s1, [t2, t3], NAME, PER)
    m2 = Mention.create(s1, [t5], NAME, MISC)
    m3 = Mention.create(s1, [t6], NAME, LOC)
    m4 = Mention.create(s1, [t7], NAME, PER)
    assert encoder.decode_mentions(s1, labels1) == [m1, m2, m3, m4]
    assert encoder.encode_mentions(s1, [m1, m2, m3, m4]) == tuple(labels1)

    labels2 = ["B-PER", "I-PER", "I-PER", "L-PER", "U-MISC", "B-LOC", "L-LOC"]
    m1 = Mention.create(s1, [t1, t2, t3, t4], NAME, PER)
    m2 = Mention.create(s1, [t5], NAME, MISC)
    m3 = Mention.create(s1, [t6, t7], NAME, LOC)
    assert encoder.decode_mentions(s1, labels2) == [m1, m2, m3]
    assert encoder.encode_mentions(s1, [m1, m2, m3]) == tuple(labels2)

    labels3 = ["O", "O", "O", "O", "O", "O", "O"]
    assert encoder.decode_mentions(s1, labels3) == []
    assert encoder.encode_mentions(s1, []) == tuple(labels3)

    # Test edge cases from bad decoding
    labels4 = ["I-PER", "B-PER", "I-ORG", "B-LOC", "U-LOC", "O", "I-MISC"]
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t2], NAME, PER)
    m3 = Mention.create(s1, [t3], NAME, ORG)
    m4 = Mention.create(s1, [t4], NAME, LOC)
    m5 = Mention.create(s1, [t5], NAME, LOC)
    m6 = Mention.create(s1, [t7], NAME, MISC)
    assert encoder.decode_mentions(s1, labels4) == [m1, m2, m3, m4, m5, m6]

    # Test edge cases from bad decoding
    labels5 = ["B-PER", "I-ORG", "I-ORG", "L-LOC", "L-PER", "B-ORG", "I-MISC"]
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t2, t3], NAME, ORG)
    m3 = Mention.create(s1, [t4], NAME, LOC)
    m4 = Mention.create(s1, [t5], NAME, PER)
    m5 = Mention.create(s1, [t6], NAME, ORG)
    m6 = Mention.create(s1, [t7], NAME, MISC)
    assert encoder.decode_mentions(s1, labels5) == [m1, m2, m3, m4, m5, m6]

    # Test edge cases from bad decoding
    labels6 = ["B-PER", "O", "I-ORG", "O", "I-LOC", "O", "B-MISC"]
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t3], NAME, ORG)
    m3 = Mention.create(s1, [t5], NAME, LOC)
    m4 = Mention.create(s1, [t7], NAME, MISC)
    assert encoder.decode_mentions(s1, labels6) == [m1, m2, m3, m4]


def test_bmes_encoder():
    encoder = BMES()

    labels1 = ["O", "B-PER", "E-PER", "O", "S-MISC", "S-LOC", "S-PER"]
    m1 = Mention.create(s1, [t2, t3], NAME, PER)
    m2 = Mention.create(s1, [t5], NAME, MISC)
    m3 = Mention.create(s1, [t6], NAME, LOC)
    m4 = Mention.create(s1, [t7], NAME, PER)
    assert encoder.decode_mentions(s1, labels1) == [m1, m2, m3, m4]
    assert encoder.encode_mentions(s1, [m1, m2, m3, m4]) == tuple(labels1)

    labels2 = ["B-PER", "M-PER", "M-PER", "E-PER", "S-MISC", "B-LOC", "E-LOC"]
    m1 = Mention.create(s1, [t1, t2, t3, t4], NAME, PER)
    m2 = Mention.create(s1, [t5], NAME, MISC)
    m3 = Mention.create(s1, [t6, t7], NAME, LOC)
    assert encoder.decode_mentions(s1, labels2) == [m1, m2, m3]
    assert encoder.encode_mentions(s1, [m1, m2, m3]) == tuple(labels2)

    labels3 = ["O", "O", "O", "O", "O", "O", "O"]
    assert encoder.decode_mentions(s1, labels3) == []
    assert encoder.encode_mentions(s1, []) == tuple(labels3)

    # Test edge cases from bad decoding
    labels4 = ["M-PER", "B-PER", "M-ORG", "B-LOC", "S-LOC", "O", "E-MISC"]
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t2], NAME, PER)
    m3 = Mention.create(s1, [t3], NAME, ORG)
    m4 = Mention.create(s1, [t4], NAME, LOC)
    m5 = Mention.create(s1, [t5], NAME, LOC)
    m6 = Mention.create(s1, [t7], NAME, MISC)
    assert encoder.decode_mentions(s1, labels4) == [m1, m2, m3, m4, m5, m6]

    # Test edge cases from bad decoding
    labels5 = ["B-PER", "M-ORG", "M-ORG", "E-LOC", "E-PER", "B-ORG", "M-MISC"]
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t2, t3], NAME, ORG)
    m3 = Mention.create(s1, [t4], NAME, LOC)
    m4 = Mention.create(s1, [t5], NAME, PER)
    m5 = Mention.create(s1, [t6], NAME, ORG)
    m6 = Mention.create(s1, [t7], NAME, MISC)
    assert encoder.decode_mentions(s1, labels5) == [m1, m2, m3, m4, m5, m6]

    # Test edge cases from bad decoding
    labels6 = ["B-PER", "O", "M-ORG", "O", "M-LOC", "O", "B-MISC"]
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t3], NAME, ORG)
    m3 = Mention.create(s1, [t5], NAME, LOC)
    m4 = Mention.create(s1, [t7], NAME, MISC)
    assert encoder.decode_mentions(s1, labels6) == [m1, m2, m3, m4]


def test_bioes_encoder():
    encoder = BIOES()

    labels1 = ["O", "B-PER", "E-PER", "O", "S-MISC", "S-LOC", "S-PER"]
    m1 = Mention.create(s1, [t2, t3], NAME, PER)
    m2 = Mention.create(s1, [t5], NAME, MISC)
    m3 = Mention.create(s1, [t6], NAME, LOC)
    m4 = Mention.create(s1, [t7], NAME, PER)
    assert encoder.decode_mentions(s1, labels1) == [m1, m2, m3, m4]
    assert encoder.encode_mentions(s1, [m1, m2, m3, m4]) == tuple(labels1)

    labels2 = ["B-PER", "I-PER", "I-PER", "E-PER", "S-MISC", "B-LOC", "E-LOC"]
    m1 = Mention.create(s1, [t1, t2, t3, t4], NAME, PER)
    m2 = Mention.create(s1, [t5], NAME, MISC)
    m3 = Mention.create(s1, [t6, t7], NAME, LOC)
    assert encoder.decode_mentions(s1, labels2) == [m1, m2, m3]
    assert encoder.encode_mentions(s1, [m1, m2, m3]) == tuple(labels2)

    labels3 = ["O", "O", "O", "O", "O", "O", "O"]
    assert encoder.decode_mentions(s1, labels3) == []
    assert encoder.encode_mentions(s1, []) == tuple(labels3)

    # Test edge cases from bad decoding
    labels4 = ["I-PER", "B-PER", "I-ORG", "B-LOC", "S-LOC", "O", "E-MISC"]
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t2], NAME, PER)
    m3 = Mention.create(s1, [t3], NAME, ORG)
    m4 = Mention.create(s1, [t4], NAME, LOC)
    m5 = Mention.create(s1, [t5], NAME, LOC)
    m6 = Mention.create(s1, [t7], NAME, MISC)
    assert encoder.decode_mentions(s1, labels4) == [m1, m2, m3, m4, m5, m6]

    # Test edge cases from bad decoding
    labels5 = ["B-PER", "I-ORG", "I-ORG", "E-LOC", "E-PER", "B-ORG", "I-MISC"]
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t2, t3], NAME, ORG)
    m3 = Mention.create(s1, [t4], NAME, LOC)
    m4 = Mention.create(s1, [t5], NAME, PER)
    m5 = Mention.create(s1, [t6], NAME, ORG)
    m6 = Mention.create(s1, [t7], NAME, MISC)
    assert encoder.decode_mentions(s1, labels5) == [m1, m2, m3, m4, m5, m6]

    # Test edge cases from bad decoding
    labels6 = ["B-PER", "O", "I-ORG", "O", "I-LOC", "O", "B-MISC"]
    m1 = Mention.create(s1, [t1], NAME, PER)
    m2 = Mention.create(s1, [t3], NAME, ORG)
    m3 = Mention.create(s1, [t5], NAME, LOC)
    m4 = Mention.create(s1, [t7], NAME, MISC)
    assert encoder.decode_mentions(s1, labels6) == [m1, m2, m3, m4]


def test_iob_encoder():
    encoder = IOB()

    labels1 = ["I-PER", "I-PER", "I-LOC", "B-LOC", "I-LOC", "I-LOC", "B-LOC"]
    m1 = Mention.create(s1, [t1, t2], NAME, PER)
    m2 = Mention.create(s1, [t3], NAME, LOC)
    m3 = Mention.create(s1, [t4, t5, t6], NAME, LOC)
    m4 = Mention.create(s1, [t7], NAME, LOC)

    assert encoder.decode_mentions(s1, labels1) == [m1, m2, m3, m4]
    assert encoder.encode_mentions(s1, [m1, m2, m3, m4]) == tuple(labels1)

    labels2 = ["O", "I-PER", "O", "I-LOC", "I-LOC", "B-LOC", "B-LOC"]
    m1 = Mention.create(s1, [t2], NAME, PER)
    m2 = Mention.create(s1, [t4, t5], NAME, LOC)
    m3 = Mention.create(s1, [t6], NAME, LOC)
    m4 = Mention.create(s1, [t7], NAME, LOC)
    assert encoder.decode_mentions(s1, labels2) == [m1, m2, m3, m4]
    assert encoder.encode_mentions(s1, [m1, m2, m3, m4]) == tuple(labels2)

    labels3 = ["O", "O", "O", "O", "O", "O", "O"]
    assert encoder.decode_mentions(s1, labels3) == []
    assert encoder.encode_mentions(s1, []) == tuple(labels3)

    # Test edge cases from bad decoding
    labels4 = ["O", "I-PER", "B-PER", "I-ORG", "O", "B-MISC", "I-LOC"]
    m1 = Mention.create(s1, [t2], NAME, PER)
    m2 = Mention.create(s1, [t3], NAME, PER)
    m3 = Mention.create(s1, [t4], NAME, ORG)
    m4 = Mention.create(s1, [t6], NAME, MISC)
    m5 = Mention.create(s1, [t7], NAME, LOC)
    assert encoder.decode_mentions(s1, labels4) == [m1, m2, m3, m4, m5]


def test_bio_encoder():
    encoder = BIO()

    labels1 = ["B-PER", "I-PER", "B-LOC", "O", "B-LOC", "I-LOC", "I-LOC"]
    m1 = Mention.create(s1, [t1, t2], NAME, PER)
    m2 = Mention.create(s1, [t3], NAME, LOC)
    m3 = Mention.create(s1, [t5, t6, t7], NAME, LOC)

    assert encoder.decode_mentions(s1, labels1) == [m1, m2, m3]
    assert encoder.encode_mentions(s1, [m1, m2, m3]) == tuple(labels1)

    labels2 = ["O", "B-PER", "I-PER", "B-PER", "O", "B-MISC", "B-LOC"]
    m1 = Mention.create(s1, [t2, t3], NAME, PER)
    m2 = Mention.create(s1, [t4], NAME, PER)
    m3 = Mention.create(s1, [t6], NAME, MISC)
    m4 = Mention.create(s1, [t7], NAME, LOC)
    assert encoder.decode_mentions(s1, labels2) == [m1, m2, m3, m4]
    assert encoder.encode_mentions(s1, [m1, m2, m3, m4]) == tuple(labels2)

    labels3 = ["O", "O", "O", "O", "O", "O", "O"]
    assert encoder.decode_mentions(s1, labels3) == []
    assert encoder.encode_mentions(s1, []) == tuple(labels3)

    # Test edge cases from bad decoding
    labels4 = ["O", "I-PER", "B-PER", "I-ORG", "O", "B-MISC", "I-LOC"]
    m1 = Mention.create(s1, [t2], NAME, PER)
    m2 = Mention.create(s1, [t3], NAME, PER)
    m3 = Mention.create(s1, [t4], NAME, ORG)
    m4 = Mention.create(s1, [t6], NAME, MISC)
    m5 = Mention.create(s1, [t7], NAME, LOC)
    assert encoder.decode_mentions(s1, labels4) == [m1, m2, m3, m4, m5]


def test_get_mention_encoder():
    assert get_mention_encoder("IO") == IO
    assert get_mention_encoder("IOB") == IOB
    assert get_mention_encoder("IOB1") == IOB
    assert get_mention_encoder("BIO") == BIO
    assert get_mention_encoder("IOB2") == BIO
    assert get_mention_encoder("BILOU") == BILOU
    assert get_mention_encoder("BIOES") == BIOES
    assert get_mention_encoder("BMES") == BMES
    assert get_mention_encoder("IOBES") == BIOES
    with pytest.raises(ValueError):
        get_mention_encoder("unknown")
