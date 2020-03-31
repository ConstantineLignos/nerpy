import io

import pytest

from nerpy import EntityType, OntoNotesIngester

short1 = io.StringIO(
    """<DOC DOCNO="nw/wsj/00/wsj_0005@0005@wsj@nw@en@on">
    <ENAMEX TYPE="PERSON">J.P. Bolduc</ENAMEX> , vice chairman of <ENAMEX TYPE="ORG">W.R. Grace -AMP- Co.</ENAMEX> , which holds a <ENAMEX TYPE="PERCENT">83.4 %</ENAMEX> interest in this energy - services company , was elected a director .
    He succeeds <ENAMEX TYPE="PERSON">Terrence D. Daniels</ENAMEX> , formerly a <ENAMEX TYPE="ORG">W.R. Grace</ENAMEX> vice chairman , who resigned .
    <ENAMEX TYPE="ORG">W.R. Grace</ENAMEX> holds <ENAMEX TYPE="CARDINAL">three</ENAMEX> of <ENAMEX TYPE="ORG">Grace Energy \'s</ENAMEX> <ENAMEX TYPE="CARDINAL">seven</ENAMEX> board seats .
    </DOC>"""
)

# To test skipping of empty line and doc id extraction
short2 = io.StringIO(
    """<DOC DOCNO="nw/wsj/00/wsj_0005@0005@wsj@nw@en@on">
    \n
    </DOC>"""
)

PERSON = EntityType("PERSON")
ORG = EntityType("ORG")
PERCENT = EntityType("PERCENT")
CARDINAL = EntityType("CARDINAL")


SHORT1_SENTENCES = [
    '<ENAMEX TYPE="PERSON">J.P. Bolduc</ENAMEX> , vice chairman of <ENAMEX TYPE="ORG">W.R. Grace -AMP- Co.</ENAMEX> , which holds a <ENAMEX TYPE="PERCENT">83.4 %</ENAMEX> interest in this energy - services company , was elected a director .',
    'He succeeds <ENAMEX TYPE="PERSON">Terrence D. Daniels</ENAMEX> , formerly a <ENAMEX TYPE="ORG">W.R. Grace</ENAMEX> vice chairman , who resigned .',
    '<ENAMEX TYPE="ORG">W.R. Grace</ENAMEX> holds <ENAMEX TYPE="CARDINAL">three</ENAMEX> of <ENAMEX TYPE="ORG">Grace Energy \'s</ENAMEX> <ENAMEX TYPE="CARDINAL">seven</ENAMEX> board seats .',
]
SHORT1_TOKENS = [
    "J.P.",
    "Bolduc",
    ",",
    "vice",
    "chairman",
    "of",
    "W.R.",
    "Grace",
    "&",
    "Co.",
    ",",
    "which",
    "holds",
    "a",
    "83.4",
    "%",
    "interest",
    "in",
    "this",
    "energy",
    "-",
    "services",
    "company",
    ",",
    "was",
    "elected",
    "a",
    "director",
    ".",
    "He",
    "succeeds",
    "Terrence",
    "D.",
    "Daniels",
    ",",
    "formerly",
    "a",
    "W.R.",
    "Grace",
    "vice",
    "chairman",
    ",",
    "who",
    "resigned",
    ".",
    "W.R.",
    "Grace",
    "holds",
    "three",
    "of",
    "Grace",
    "Energy",
    "'s",
    "seven",
    "board",
    "seats",
    ".",
]
SHORT1_MENTIONS = [
    ("J.P. Bolduc", PERSON),
    ("W.R. Grace & Co.", ORG),
    ("83.4 %", PERCENT),
    ("Terrence D. Daniels", PERSON),
    ("W.R. Grace", ORG),
    ("W.R. Grace", ORG),
    ("three", CARDINAL),
    ("Grace Energy 's", ORG),
    ("seven", CARDINAL),
]

bad_enamex1 = io.StringIO(
    """<DOC DOCNO="nw/wsj/00/wsj_0005@0005@wsj@nw@en@on">
    <ENAMEX TYPE="PERSON"></ENAMEX>
    </DOC>"""
)

bad_enamex2 = io.StringIO(
    """<DOC DOCNO="nw/wsj/00/wsj_0005@0005@wsj@nw@en@on">
    <ENAMEX TYPE="PERSON">  </ENAMEX>
    </DOC>"""
)

bad_enamex3 = io.StringIO(
    """<DOC DOCNO="nw/wsj/00/wsj_0005@0005@wsj@nw@en@on">
    <ENAMEX>Bob</ENAMEX>
    </DOC>"""
)

bad_doc1 = io.StringIO(
    """<DOC>
    <ENAMEX TYPE="ORG">non-State-owned</ENAMEX>
    </DOC>"""
)

bad_doc2 = io.StringIO("")

bad_doc3 = io.StringIO("""<ENAMEX TYPE="ORG">non-State-owned</ENAMEX>""")

good_enamex1 = io.StringIO(
    """<DOC DOCNO="nw/wsj/00/wsj_0005@0005@wsj@nw@en@on">
    <ENAMEX TYPE="ORG" S_OFF="4" E_OFF="6">non-State-owned</ENAMEX>
    </DOC>"""
)

good_enamex2 = io.StringIO(
    """<DOC DOCNO="nw/wsj/00/wsj_0005@0005@wsj@nw@en@on">
    <ENAMEX TYPE="ORG" S_OFF="4">non-State-owned</ENAMEX>
    </DOC>"""
)

good_enamex3 = io.StringIO(
    """<DOC DOCNO="nw/wsj/00/wsj_0005@0005@wsj@nw@en@on">
    <ENAMEX TYPE="ORG" E_OFF="6">non-State-owned</ENAMEX>
    </DOC>"""
)

good_enamex4 = io.StringIO(
    """<DOC DOCNO="nw/wsj/00/wsj_0005@0005@wsj@nw@en@on">
    <ENAMEX TYPE="ORG">non-State-owned</ENAMEX>
    </DOC>"""
)


@pytest.fixture(scope="module")
def ingester():
    return OntoNotesIngester()


def test_basic(ingester: OntoNotesIngester) -> None:
    """Basic ingest works."""
    doc = ingester.ingest(short1, "test")
    assert doc.id == "test"
    assert [
        token.text for sentence in doc.sentences for token in sentence.tokens
    ] == SHORT1_TOKENS

    mentions = [
        (mention.tokenized_text(doc), mention.entity_type) for mention in doc.mentions
    ]
    assert mentions == SHORT1_MENTIONS


def test_e_off_s_off(ingester: OntoNotesIngester) -> None:
    """E_OFF and S_OFF have no effect."""
    for file in (good_enamex1, good_enamex2, good_enamex3, good_enamex4):
        doc = ingester.ingest(file, "test")
        mention = doc.mentions[0]
        mention_text = mention.tokenized_text(doc)
        assert mention_text == "non-State-owned"


def test_bad_name(ingester: OntoNotesIngester) -> None:
    """Bad ENAMEX tags are rejected."""
    # Empty name
    with pytest.raises(ValueError):
        ingester.ingest(bad_enamex1, "test")
    # Whitespace name
    with pytest.raises(ValueError):
        ingester.ingest(bad_enamex2, "test")
    # No type
    with pytest.raises(ValueError):
        ingester.ingest(bad_enamex3, "test")


def test_bad_doc(ingester: OntoNotesIngester) -> None:
    """Bad documents are rejected."""
    # No DOCNAME
    with pytest.raises(ValueError):
        ingester.ingest(bad_doc1)

    # Empty doc
    with pytest.raises(ValueError):
        ingester.ingest(bad_doc2)

    # No opening doc tag
    with pytest.raises(ValueError):
        ingester.ingest(bad_doc3)


def test_replace_punc() -> None:
    """Treebank-style punctation replacements are handled."""
    ingester = OntoNotesIngester()
    assert ingester._token_text("foo-bar-baz") == "foo-bar-baz"
    assert ingester._token_text("-RRB-") == ")"
    assert ingester._token_text("-AMP-") == "&"
    assert ingester._token_text("-LAB-http://isi.edu/-RAB-") == "<http://isi.edu/>"


def test_docid_extraction(ingester: OntoNotesIngester) -> None:
    # Test doc id extraction
    doc = ingester.ingest(short2)
    assert doc.id == "0005_wsj_nw_en_on"
