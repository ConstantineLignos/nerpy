import tempfile

from nerpy import DocumentBuilder, Token
from nerpy.io import load_json, load_pickled_documents, pickle_documents


def test_pickling():
    builder = DocumentBuilder("test")
    t1 = Token("foo", 0)
    t2 = Token("bar", 1)
    builder.create_sentence([t1, t2])
    doc = builder.build()

    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = tmpdirname + "/tmpfile"
        pickle_documents([doc], filename)
        pickled_docs = load_pickled_documents(filename)

        assert pickled_docs == [doc]


def test_json():
    json = load_json("tests/test_data/test_json.json")
    assert len(json.keys()) == 2
