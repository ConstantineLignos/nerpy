import tempfile
from typing import Mapping

from nerpy import BILOU, BIO, DocumentBuilder, EntityType, Mention, MentionType, Token
from nerpy.annotators.crfsuite import CRFSuiteAnnotator, train_crfsuite
from nerpy.features import SentenceFeatureExtractor

NAME = MentionType("name")
ORG = EntityType(types=("ORG",))
PER = EntityType(types=("PER",))
LOC = EntityType(types=("LOC",))
MISC = EntityType(types=("MISC",))


def test_annotator():
    # Create sample document
    builder = DocumentBuilder("test")
    t0 = Token("EU", 0)
    t1 = Token("rejects", 1)
    t2 = Token("German", 2)
    t3 = Token("call", 3)
    t4 = Token("to", 4)
    t5 = Token("boycott", 5)
    t6 = Token("British", 6)
    t7 = Token("lamb", 7)
    t8 = Token(".", 8)
    s1 = builder.create_sentence([t0, t1, t2, t3, t4, t5, t6, t7, t8])
    m0 = Mention.create(s1, [t0], NAME, ORG)
    m1 = Mention.create(s1, [t2], NAME, MISC)
    m2 = Mention.create(s1, [t6], NAME, MISC)
    builder.add_mentions([m0, m1, m2])
    doc = builder.build()

    # Set params
    train_params = {"max_iterations": 100}
    feature_params = {"baseline": {"window": [-2, -1, 0, 1, 2], "token_identity": {}}}

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Test training
        annotator1 = _create_annotator(feature_params)
        model_path = tmpdirname + "/model"
        annotator1.train(
            [doc],
            model_path=model_path,
            algorithm="ap",
            train_params=train_params,
            verbose=False,
            log_file=None,
        )
        pred_doc = annotator1.add_mentions(doc.copy_without_mentions())
        assert pred_doc.mentions == (m0, m1, m2)

        # Test saved model
        saved_annotator = CRFSuiteAnnotator.from_model(
            MentionType("name"),
            SentenceFeatureExtractor(feature_params),
            BILOU(),
            model_path,
        )
        pred_doc = saved_annotator.add_mentions(doc.copy_without_mentions())
        assert pred_doc.mentions == (m0, m1, m2)

        # Test empty train params
        annotator2 = _create_annotator(feature_params)
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = tmpdirname + "/model"
            annotator2.train(
                [doc],
                model_path=model_path,
                algorithm="ap",
                train_params=None,
                verbose=False,
                log_file=None,
            )
        pred_doc = annotator2.add_mentions(doc.copy_without_mentions())
        assert pred_doc.mentions == (m0, m1, m2)

        # Test separate feature extraction
        annotator3 = _create_annotator(feature_params)
        features = annotator3.extract_features([doc])
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = tmpdirname + "/model"
            annotator3.train_featurized(
                features,
                model_path=model_path,
                algorithm="ap",
                train_params=None,
                verbose=False,
                log_file=None,
            )
        pred_doc = annotator3.add_mentions(doc.copy_without_mentions())
        assert pred_doc.mentions == (m0, m1, m2)

        # Test training wrapper
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = tmpdirname + "/model"
            train_params_with_alg = dict(train_params)
            train_params_with_alg["algorithm"] = "ap"
            mention_encoder = BIO()
            feature_extractor = SentenceFeatureExtractor(feature_params)
            mention_type = MentionType("name")

            annotator4 = train_crfsuite(
                mention_encoder,
                feature_extractor,
                mention_type,
                model_path,
                [doc],
                train_params_with_alg,
            )

        assert annotator4.mention_encoder == mention_encoder
        assert annotator4.feature_extractor == feature_extractor

        pred_doc = annotator4.add_mentions(doc.copy_without_mentions())
        assert pred_doc.mentions == (m0, m1, m2)


def _create_annotator(feature_params: Mapping) -> CRFSuiteAnnotator:
    return CRFSuiteAnnotator.for_training(
        MentionType("name"), SentenceFeatureExtractor(feature_params), BILOU()
    )
