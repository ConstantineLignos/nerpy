import tempfile

from nerpy import BILOU, DocumentBuilder, EntityType, Mention, MentionType, Token
from nerpy.annotators.seqmodels import SequenceModelsAnnotator, train_seqmodels
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
    train_params = {"max_iterations": 100, "averaged": False}
    feature_params = {"baseline": {"window": [-2, -1, 0, 1, 2], "token_identity": {}}}

    mention_type = MentionType("name")
    feature_extractor = SentenceFeatureExtractor(feature_params)
    mention_encoder = BILOU()

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Test training
        model_path = tmpdirname + "/model"
        annotator1 = train_seqmodels(
            mention_encoder,
            feature_extractor,
            mention_type,
            model_path,
            [doc],
            train_params,
            verbose=True,
        )
        assert annotator1.mention_encoder == mention_encoder
        assert annotator1.feature_extractor == feature_extractor

        pred_doc = annotator1.add_mentions(doc.copy_without_mentions())
        assert pred_doc.mentions == (m0, m1, m2)

        # Test saved model
        saved_annotator = SequenceModelsAnnotator.from_model(
            MentionType("name"),
            SentenceFeatureExtractor(feature_params),
            BILOU(),
            model_path,
        )
        pred_doc = saved_annotator.add_mentions(doc.copy_without_mentions())
        assert pred_doc.mentions == (m0, m1, m2)
