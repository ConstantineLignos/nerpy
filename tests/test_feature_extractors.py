import pytest

from nerpy import DocumentBuilder, Token
from nerpy.features import (
    POS,
    AllCaps,
    AllNumeric,
    BrownClusterFeatures,
    ContainsNumber,
    IsCapitalized,
    IsPunc,
    LengthValue,
    LengthWeight,
    Prefix,
    SentenceFeatureExtractor,
    Suffix,
    TokenIdentity,
    WordEmbeddingFeatures,
    WordShape,
)


def test_word_vectors():
    t0 = Token("the", 0)
    t1 = Token("Wikipedia", 1)
    t2 = Token("article", 2)
    t3 = Token(".", 3)
    t4 = Token("foobar", 4)

    token_features = {}

    extractor = WordEmbeddingFeatures("tests/test_data/word_vectors.sqlite")
    extractor.extract(t0, -1, token_features)
    extractor.extract(t1, 0, token_features)
    extractor.extract(t2, 1, token_features)

    # Vectors
    # the 0.0129 0.0026 0.0098
    # Wikipedia 0.0007 -0.0205 0.0107
    # article 0.0050 -0.0114 0.0150
    assert token_features["v[-1]=0"] == pytest.approx(0.0129)
    assert token_features["v[-1]=1"] == pytest.approx(0.0026)
    assert token_features["v[-1]=2"] == pytest.approx(0.0098)
    assert token_features["v[0]=0"] == pytest.approx(0.0007)
    assert token_features["v[0]=1"] == pytest.approx(-0.0205)
    assert token_features["v[0]=2"] == pytest.approx(0.0107)
    assert token_features["v[1]=0"] == pytest.approx(0.0050)
    assert token_features["v[1]=1"] == pytest.approx(-0.0114)
    assert token_features["v[1]=2"] == pytest.approx(0.0150)

    # Punctuation should not have any feature added
    token_features = {}
    extractor.extract(t3, 0, token_features)
    assert len(token_features) == 0

    # Token not in word vectors file
    token_features = {}
    extractor.extract(t4, 0, token_features)
    assert token_features == {"v[0]=OOV": 1.0}

    # Check scaling
    extractor = WordEmbeddingFeatures("tests/test_data/word_vectors.sqlite", scale=2.0)
    token_features = {}
    extractor.extract(t0, 0, token_features)
    assert token_features["v[0]=0"] == pytest.approx(0.0129 * 2.0)
    assert token_features["v[0]=1"] == pytest.approx(0.0026 * 2.0)
    assert token_features["v[0]=2"] == pytest.approx(0.0098 * 2.0)


def test_brown_clusters():
    clusters_path = "tests/test_data/test_clusters.paths"
    t0 = Token("the", 0)
    t1 = Token("Wikipedia", 1)
    t2 = Token("article", 2)

    # Test full paths
    # 10100   the     3241457
    # 110110100110110100      article 2038
    token_features = {}
    extractor = BrownClusterFeatures(clusters_path, use_full_paths=True)
    extractor.extract(t0, -1, token_features)
    extractor.extract(t1, 0, token_features)
    extractor.extract(t2, 1, token_features)
    assert token_features == {
        "bc[-1]=10100": 1.0,
        "bc[0]=OOV": 1.0,
        "bc[1]=110110100110110100": 1.0,
    }

    # Test specific prefixes
    token_features = {}
    extractor = BrownClusterFeatures(clusters_path, use_prefixes=True, prefixes=[1, 2])
    extractor.extract(t0, -1, token_features)
    extractor.extract(t1, 0, token_features)
    extractor.extract(t2, 1, token_features)
    assert token_features == {
        "bc[-1]=1": 1.0,
        "bc[-1]=10": 1.0,
        "bc[0]=OOV": 1.0,
        "bc[1]=1": 1.0,
        "bc[1]=11": 1.0,
    }

    # Test all prefixes
    token_features = {}
    extractor = BrownClusterFeatures(clusters_path, use_prefixes=True)
    extractor.extract(t0, -1, token_features)
    extractor.extract(t1, 0, token_features)
    extractor.extract(t2, 1, token_features)
    assert token_features == {
        "bc[-1]=1": 1.0,
        "bc[-1]=10": 1.0,
        "bc[-1]=101": 1.0,
        "bc[-1]=1010": 1.0,
        "bc[-1]=10100": 1.0,
        "bc[0]=OOV": 1.0,
        "bc[1]=1": 1.0,
        "bc[1]=11": 1.0,
        "bc[1]=110": 1.0,
        "bc[1]=1101": 1.0,
        "bc[1]=11011": 1.0,
        "bc[1]=110110": 1.0,
        "bc[1]=1101101": 1.0,
        "bc[1]=11011010": 1.0,
        "bc[1]=110110100": 1.0,
        "bc[1]=1101101001": 1.0,
        "bc[1]=11011010011": 1.0,
        "bc[1]=110110100110": 1.0,
        "bc[1]=1101101001101": 1.0,
        "bc[1]=11011010011011": 1.0,
        "bc[1]=110110100110110": 1.0,
        "bc[1]=1101101001101101": 1.0,
        "bc[1]=11011010011011010": 1.0,
        "bc[1]=110110100110110100": 1.0,
    }

    # Test bad usage
    with pytest.raises(ValueError):
        # No features
        BrownClusterFeatures(clusters_path)

    with pytest.raises(ValueError):
        # Prefixes without use_prefixes
        BrownClusterFeatures(clusters_path, use_full_paths=True, prefixes=[1])

    with pytest.raises(ValueError):
        # Full paths and all prefixes
        BrownClusterFeatures(clusters_path, use_full_paths=True, use_prefixes=True)

    with pytest.raises(ValueError):
        # Empty prefixes
        BrownClusterFeatures(clusters_path, use_prefixes=True, prefixes=[])

    with pytest.raises(ValueError):
        # Zero isn't a valid prefix
        BrownClusterFeatures(clusters_path, use_prefixes=True, prefixes=[0])

    with pytest.raises(ValueError):
        # Prefixes must be ints
        BrownClusterFeatures(clusters_path, use_prefixes=True, prefixes=[1.0])

    with pytest.raises(ValueError):
        # Bad lines in file
        BrownClusterFeatures(
            "tests/test_data/bad_test_clusters.paths", use_full_paths=True
        )


def test_token_identity():
    t0 = Token("Foo", 0)

    token_features = {}
    extractor = TokenIdentity()
    extractor.extract(t0, 0, token_features)
    assert token_features == {"tkn[0]=Foo": 1.0}

    token_features = {}
    extractor = TokenIdentity(lowercase=True)
    extractor.extract(t0, 0, token_features)
    assert token_features == {"tkn[0]=foo": 1.0}


def test_capitalized():
    t0 = Token("Foo", 0)
    t1 = Token("foo", 1)
    t2 = Token("fOo", 2)

    token_features = {}

    extractor = IsCapitalized()
    extractor.extract(t0, 0, token_features)
    extractor.extract(t1, 1, token_features)
    extractor.extract(t2, 2, token_features)

    assert token_features == {"cap[0]=True": 1.0}


def test_is_punc():
    t0 = Token(".", 0)
    t1 = Token('"', 1)
    t2 = Token("foo", 2)

    token_features = {}

    extractor = IsPunc()
    extractor.extract(t0, 0, token_features)
    extractor.extract(t1, 1, token_features)
    extractor.extract(t2, 2, token_features)

    assert token_features == {"punc[0]=True": 1.0, "punc[1]=True": 1.0}


def test_all_caps():
    t0 = Token("FOO", 0)
    t1 = Token("foo", 1)
    t2 = Token("Foo", 2)

    token_features = {}

    extractor = AllCaps()
    extractor.extract(t0, 0, token_features)
    extractor.extract(t1, 1, token_features)
    extractor.extract(t2, 2, token_features)

    assert token_features == {"all_caps[0]=True": 1.0}


def test_all_numeric():
    t0 = Token("123", 0)
    t1 = Token("10.23", 1)
    t2 = Token(".29", 2)
    t3 = Token("10,000,000", 3)
    t4 = Token(".", 4)
    t5 = Token("Foo1", 5)

    token_features = {}

    extractor = AllNumeric()
    extractor.extract(t0, 0, token_features)
    extractor.extract(t1, 1, token_features)
    extractor.extract(t2, 2, token_features)
    extractor.extract(t3, 3, token_features)
    extractor.extract(t4, 4, token_features)
    extractor.extract(t5, 5, token_features)

    assert token_features == {
        "all_num[0]=True": 1.0,
        "all_num[1]=True": 1.0,
        "all_num[2]=True": 1.0,
        "all_num[3]=True": 1.0,
    }


def test_contains_number():
    t0 = Token("10", 0)
    t1 = Token("DC10-30", 1)
    t2 = Token("foo", 2)

    token_features = {}

    extractor = ContainsNumber()
    extractor.extract(t0, 0, token_features)
    extractor.extract(t1, 1, token_features)
    extractor.extract(t2, 2, token_features)

    assert token_features == {
        "cntns_num[0]=True": 1.0,
        "cntns_num[1]=True": 1.0,
    }


def test_length_value():
    t0 = Token("foo", 0)
    t1 = Token(".", 1)

    token_features = {}

    extractor = LengthValue()
    extractor.extract(t0, 0, token_features)
    extractor.extract(t1, 1, token_features)

    assert token_features == {"len_val[0]=3": 1.0, "len_val[1]=1": 1.0}


def test_length_weight():
    t0 = Token("foo", 0)
    t1 = Token(".", 1)

    token_features = {}

    extractor = LengthWeight()
    extractor.extract(t0, 0, token_features)
    extractor.extract(t1, 1, token_features)

    assert token_features == {"len_weight[0]": 3, "len_weight[1]": 1}


def test_pos():
    t0 = Token.create("foo", 0, pos_tag="NNP")
    t1 = Token(".", 1)

    token_features = {}

    extractor = POS()
    extractor.extract(t0, 0, token_features)
    extractor.extract(t1, 1, token_features)

    assert token_features == {"pos[0]=NNP": 1.0}


def test_sfx():
    t0 = Token("HelloWorld", 0)
    t1 = Token("a", 1)
    t2 = Token("the", 2)

    token_features = {}
    feature_kwargs = {"min_length": 1, "max_length": 3}

    extractor = Suffix(**feature_kwargs)
    extractor.extract(t0, 0, token_features)
    extractor.extract(t1, 1, token_features)
    extractor.extract(t2, 2, token_features)

    assert token_features == {
        "sfx[0]=d": 1.0,
        "sfx[0]=ld": 1.0,
        "sfx[0]=rld": 1.0,
        "sfx[1]=a": 1.0,
        "sfx[2]=e": 1.0,
        "sfx[2]=he": 1.0,
        "sfx[2]=the": 1.0,
    }


def test_pfx():
    t0 = Token("HelloWorld", 0)
    t1 = Token("a", 1)
    t2 = Token("the", 2)

    token_features = {}
    feature_kwargs = {"min_length": 1, "max_length": 3}

    extractor = Prefix(**feature_kwargs)
    extractor.extract(t0, 0, token_features)
    extractor.extract(t1, 1, token_features)
    extractor.extract(t2, 2, token_features)

    assert token_features == {
        "pfx[0]=H": 1.0,
        "pfx[0]=He": 1.0,
        "pfx[0]=Hel": 1.0,
        "pfx[1]=a": 1.0,
        "pfx[2]=t": 1.0,
        "pfx[2]=th": 1.0,
        "pfx[2]=the": 1.0,
    }


def test_shape():
    t0 = Token("08-20-2019", 0)
    t1 = Token("111-111-1111", 1)
    t2 = Token("U.S.A", 2)
    t3 = Token("DC10-30", 3)
    t4 = Token("Foo", 4)

    token_features = {}

    extractor = WordShape()
    extractor.extract(t0, 0, token_features)
    extractor.extract(t1, 1, token_features)
    extractor.extract(t2, 2, token_features)
    extractor.extract(t3, 3, token_features)
    extractor.extract(t4, 4, token_features)

    assert token_features == {
        "shape[0]=00-00-0000": 1.0,
        "shape[1]=000-000-0000": 1.0,
        "shape[2]=A.A.A": 1.0,
        "shape[3]=AA00-00": 1.0,
        "shape[4]=Aaa": 1.0,
    }


def test_feature_extraction():
    feature_params = {
        "baseline": {"window": [-1, 0, 1], "bias": {}, "token_identity": {}}
    }
    feature_extractor = SentenceFeatureExtractor(feature_params)
    builder = DocumentBuilder("test")
    t0 = Token("foo", 0)
    t1 = Token("bar", 1)
    t2 = Token("foobar", 2)
    t3 = Token("foobaz", 3)
    t4 = Token("barbaz", 4)
    s1 = builder.create_sentence([t0, t1, t2, t3, t4])
    d = builder.build()

    features = feature_extractor.extract(s1, d)
    assert features == [
        {"b[0]": 1.0, "tkn[0]=foo": 1.0, "tkn[1]=bar": 1.0},
        {"b[0]": 1.0, "tkn[-1]=foo": 1.0, "tkn[0]=bar": 1.0, "tkn[1]=foobar": 1.0},
        {"b[0]": 1.0, "tkn[-1]=bar": 1.0, "tkn[0]=foobar": 1.0, "tkn[1]=foobaz": 1.0},
        {"b[0]": 1.0, "tkn[-1]=foobar": 1.0, "tkn[0]=foobaz": 1.0, "tkn[1]=barbaz": 1.0},
        {"b[0]": 1.0, "tkn[-1]=foobaz": 1.0, "tkn[0]=barbaz": 1.0},
    ]


def test_bad_feature():
    feature_params = {"baseline": {"window": [-1, 0, 1], "foo": {}}}
    with pytest.raises(ValueError):
        SentenceFeatureExtractor(feature_params)
