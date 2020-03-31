import pytest

from nerpy import DocumentBuilder, EntityType, Mention, MentionType, Token
from nerpy.scoring import ScoringCounts, score_prf

NAME = MentionType("name")
ORG = EntityType(types=("ORG",))
PER = EntityType(types=("PER",))
LOC = EntityType(types=("LOC",))
MISC = EntityType(types=("MISC",))


def create_system_doc():
    builder = DocumentBuilder("test")
    t0 = Token("foo", 0)
    t1 = Token("bar", 1)
    t2 = Token("baz", 2)
    s1 = builder.create_sentence([t0, t1, t2])

    m0 = Mention.create(s1, [t0], NAME, PER)
    m1 = Mention.create(s1, [t1], NAME, ORG)
    m2 = Mention.create(s1, [t1, t2], NAME, LOC)
    m3 = Mention.create(s1, [t0, t1, t2], NAME, LOC)
    builder.add_mentions([m0, m1, m2, m3])
    system_doc = builder.build()
    return system_doc


def create_gold_doc():
    builder = DocumentBuilder("test")
    t0 = Token("foo", 0)
    t1 = Token("bar", 1)
    t2 = Token("baz", 2)
    s1 = builder.create_sentence([t0, t1, t2])

    m0 = Mention.create(s1, [t0], NAME, PER)
    m1 = Mention.create(s1, [t1], NAME, ORG)
    m2 = Mention.create(s1, [t2], NAME, PER)
    m3 = Mention.create(s1, [t1, t2], NAME, ORG)
    m4 = Mention.create(s1, [t0, t1, t2], NAME, LOC)
    builder.add_mentions([m0, m1, m2, m3, m4])
    gold_doc = builder.build()
    return gold_doc


def test_scoring():
    system_doc = create_system_doc()
    gold_doc = create_gold_doc()

    res = score_prf([gold_doc], [system_doc])

    # Overall scores
    assert res.score.precision == 0.75
    assert res.score.recall == 0.6
    assert res.score.fscore == 0.6666666666666665

    # Per entity scores
    assert res.type_scores[PER].precision == 1.0
    assert res.type_scores[PER].recall == 0.5
    assert res.type_scores[PER].fscore == 0.6666666666666666

    assert res.type_scores[ORG].precision == 1.0
    assert res.type_scores[ORG].recall == 0.5
    assert res.type_scores[ORG].fscore == 0.6666666666666666

    assert res.type_scores[LOC].precision == 0.5
    assert res.type_scores[LOC].recall == 1.0
    assert res.type_scores[LOC].fscore == 0.6666666666666666


def test_scoring_counts():
    system_doc = create_system_doc()
    gold_doc = create_gold_doc()

    scoring_counts = ScoringCounts().count([system_doc], [gold_doc])

    # TP
    assert scoring_counts[PER]["foo"].true_positives == 1
    assert scoring_counts[ORG]["bar"].true_positives == 1
    assert scoring_counts[LOC]["foo bar baz"].true_positives == 1

    # FP
    assert scoring_counts[LOC]["bar baz"].false_positives == 1

    # FN
    assert scoring_counts[PER]["baz"].false_negatives == 1
    assert scoring_counts[ORG]["bar baz"].false_negatives == 1


def test_empty_mentions():
    gold_doc = create_gold_doc()

    # Create system doc with no mentions
    builder = DocumentBuilder("test")
    t0 = Token("foo", 0)
    t1 = Token("bar", 1)
    t2 = Token("baz", 2)
    builder.create_sentence([t0, t1, t2])
    system_doc = builder.build()

    res = score_prf([gold_doc], [system_doc])

    # Overall scores
    assert res.score.precision == 0.0
    assert res.score.recall == 0.0
    assert res.score.fscore == 0.0

    # Per entity scores
    for entity in res.type_scores:
        assert res.type_scores[entity].precision == 0.0
        assert res.type_scores[entity].recall == 0.0
        assert res.type_scores[entity].fscore == 0.0


def test_wrong_mentions():
    gold_doc = create_gold_doc()

    # Create system doc with wrong mentions
    builder = DocumentBuilder("test")
    t0 = Token("foo", 0)
    t1 = Token("bar", 1)
    t2 = Token("baz", 2)
    s1 = builder.create_sentence([t0, t1, t2])
    m0 = Mention.create(s1, [t0], NAME, MISC)
    m1 = Mention.create(s1, [t0, t1], NAME, ORG)
    m2 = Mention.create(s1, [t1, t2], NAME, MISC)
    m3 = Mention.create(s1, [t0, t1, t2], NAME, MISC)
    builder.add_mentions([m0, m1, m2, m3])
    system_doc = builder.build()

    res = score_prf([gold_doc], [system_doc])

    # Overall scores
    assert res.score.precision == 0.0
    assert res.score.recall == 0.0
    assert res.score.fscore == 0.0

    # Per entity scores
    for entity in res.type_scores:
        assert res.type_scores[entity].precision == 0.0
        assert res.type_scores[entity].recall == 0.0
        assert res.type_scores[entity].fscore == 0.0


def test_wrong_docid():
    builder = DocumentBuilder("system")
    t0 = Token("foo", 0)
    t1 = Token("bar", 1)
    t2 = Token("baz", 2)
    builder.create_sentence([t0, t1, t2])
    system_doc = builder.build()

    builder = DocumentBuilder("gold")
    t0 = Token("foo", 0)
    t1 = Token("bar", 1)
    t2 = Token("baz", 2)
    builder.create_sentence([t0, t1, t2])
    gold_doc = builder.build()

    with pytest.raises(ValueError):
        score_prf([gold_doc], [system_doc], check_docids=True)

    with pytest.raises(ValueError):
        ScoringCounts().count([system_doc], [gold_doc], check_docids=True)


def test_display():
    """ Print scoring results doesn't crash. """
    builder = DocumentBuilder("test")
    t0 = Token("foo", 0)
    t1 = Token("bar", 1)
    t2 = Token("baz", 2)
    s1 = builder.create_sentence([t0, t1, t2])
    m0 = Mention.create(s1, [t0], NAME, MISC)
    m1 = Mention.create(s1, [t0, t1], NAME, ORG)
    m2 = Mention.create(s1, [t1, t2], NAME, MISC)
    m3 = Mention.create(s1, [t0, t1, t2], NAME, MISC)
    builder.add_mentions([m0, m1, m2, m3])
    system_doc = builder.build()

    builder = DocumentBuilder("test")
    t0 = Token("foo", 0)
    t1 = Token("bar", 1)
    t2 = Token("baz", 2)
    s1 = builder.create_sentence([t0, t1, t2])
    m0 = Mention.create(s1, [t0], NAME, MISC)
    m1 = Mention.create(s1, [t0, t1], NAME, ORG)
    m2 = Mention.create(s1, [t1, t2], NAME, MISC)
    m3 = Mention.create(s1, [t0, t1, t2], NAME, MISC)
    builder.add_mentions([m0, m1, m2, m3])
    gold_doc = builder.build()

    res = score_prf([gold_doc], [system_doc])
    res.print()
