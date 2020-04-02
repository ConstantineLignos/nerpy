"""A CRFSuite-based mention annotator."""
import os
import pickle
import tempfile
import time
from os import PathLike
from pathlib import Path
from typing import IO, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from attr import attrib, attrs
from attr.validators import instance_of, optional
from pycrfsuite import Tagger, Trainer  # pylint: disable=no-name-in-module

from nerpy.annotator import SequenceMentionAnnotator, Trainable
from nerpy.document import Document, Mention, MentionType
from nerpy.encoding import MentionEncoder
from nerpy.features import ExtractedFeatures, SentenceFeatureExtractor

# TODO: Figure out how to serialize models with their strategies and feature extractors
# TODO: Refactor to reduce redundancy around feature extraction and multiple training methods


# Due to the Tagger object, cannot be frozen
@attrs
class CRFSuiteAnnotator(SequenceMentionAnnotator, Trainable):
    _mention_type: MentionType = attrib(validator=instance_of(MentionType))
    _feature_extractor: SentenceFeatureExtractor = attrib(
        validator=instance_of(SentenceFeatureExtractor)
    )
    # Mypy raised a false positive about a concrete class being needed
    _mention_encoder: MentionEncoder = attrib(
        validator=instance_of(MentionEncoder)  # type: ignore
    )
    _tagger: Tagger = attrib(validator=instance_of(Tagger))
    _tagger_bytes: Optional[bytes] = attrib(validator=optional(instance_of(bytes)))

    def __getstate__(self) -> dict:
        if self._tagger_bytes is None:
            raise ValueError(
                "Tagger does not have a binary model and cannot be serialized"
            )

        state = dict(self.__dict__)
        state["_tagger"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._tagger = Tagger()
        self._tagger.open_inmemory(self._tagger_bytes)

    @classmethod
    def for_training(
        cls,
        mention_type: MentionType,
        feature_extractor: Optional[SentenceFeatureExtractor],
        mention_encoder: MentionEncoder,
    ) -> "CRFSuiteAnnotator":
        tagger = Tagger()
        return cls(mention_type, feature_extractor, mention_encoder, tagger, None)

    @classmethod
    def from_path(cls, path: Union[str, PathLike]) -> "CRFSuiteAnnotator":
        with open(path, "rb") as file:
            return pickle.load(file)

    def to_path(self, path: Union[str, PathLike]) -> None:
        with open(path, "wb") as file:
            return pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, buf: bytes) -> "CRFSuiteAnnotator":
        return pickle.loads(buf)

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    def mentions(self, doc: Document) -> Sequence[Mention]:
        mentions: List[Mention] = []
        for sentence in doc.sentences:
            sent_x = self._feature_extractor.extract(sentence, doc)
            pred_y = self._tagger.tag(sent_x)
            mentions.extend(self._mention_encoder.decode_mentions(sentence, pred_y))

        return mentions

    @property
    def mention_encoder(self) -> MentionEncoder:
        return self._mention_encoder

    @property
    def feature_extractor(self) -> SentenceFeatureExtractor:
        return self._feature_extractor

    # Training method specifies values for the kwargs, so it will not exactly match the interface
    # noinspection PyMethodOverriding
    def train(  # type: ignore
        self,
        docs: Iterable[Document],
        *,
        tmp_model_path: Optional[Union[str, Path]] = None,
        algorithm: str,
        train_params: Optional[Mapping] = None,
        verbose: bool = False,
        log_file: Optional[IO[str]] = None,
    ) -> None:
        if train_params is None:
            train_params = {}
        trainer = Trainer(algorithm=algorithm, params=train_params, verbose=verbose)
        if tmp_model_path:
            Path(tmp_model_path).parent.mkdir(parents=True, exist_ok=True)

        mention_count = 0
        token_count = 0
        document_count = 0
        sentence_count = 0
        print("Extracting features", file=log_file)
        start_time = time.perf_counter()
        for doc in docs:
            for sentence, mentions in doc.sentences_with_mentions():
                sent_x = self._feature_extractor.extract(sentence, doc)
                sent_y = self._mention_encoder.encode_mentions(sentence, mentions)
                assert len(sent_x) == len(sent_y)
                trainer.append(sent_x, sent_y)

                mention_count += len(mentions)
                token_count += len(sent_x)
                sentence_count += 1

            document_count += 1

        print(
            "Feature extraction took {} seconds".format(time.perf_counter() - start_time),
            file=log_file,
        )
        print(
            f"Extracted features for {document_count} documents, {sentence_count} sentences, "
            f"{token_count} tokens, {mention_count} mentions",
            file=log_file,
        )

        # Set up model path
        if tmp_model_path:
            tmpdir = None
        else:
            # We need to use a temporary directory since NamedTemporaryFile cannot
            # guarantee that the file can be opened a second time.
            tmpdir = tempfile.TemporaryDirectory()
            tmp_model_path = os.path.join(tmpdir.name, "crfsuite_annotator_tmp.model")

        # Train
        print("Training", file=log_file)
        start_time = time.perf_counter()
        # Convert path to str since it can't take a PathLike
        trainer.train(str(tmp_model_path))
        print(
            "Training took {} seconds".format(time.perf_counter() - start_time),
            file=log_file,
        )

        self._tagger.open(tmp_model_path)
        with open(tmp_model_path, "rb") as model_file:
            self._tagger_bytes = model_file.read()

        if tmpdir:
            tmpdir.cleanup()

    def train_featurized(
        self,
        training_data: ExtractedFeatures,
        model_path: Union[str, Path],
        *,
        algorithm: str,
        train_params: Optional[Mapping] = None,
        verbose: bool = False,
        log_file: Optional[IO[str]] = None,
    ) -> None:
        assert (
            training_data.extractor == self._feature_extractor
        ), "Training data feature extractor differs from instance feature extractor"

        if train_params is None:
            train_params = {}
        trainer = Trainer(algorithm=algorithm, params=train_params, verbose=verbose)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        for sent_x, sent_y in zip(training_data.features, training_data.labels):
            trainer.append(sent_x, sent_y)

        start_time = time.perf_counter()
        trainer.train(model_path)
        print(
            "Training took {} seconds".format(time.perf_counter() - start_time),
            file=log_file,
        )
        self._tagger.open(model_path)


def train_crfsuite(
    mention_encoder: MentionEncoder,
    feature_extractor: SentenceFeatureExtractor,
    mention_type: MentionType,
    train_docs: Iterable[Document],
    train_params: Dict,
    *,
    tmp_model_path: Union[str, Path] = None,
    verbose: bool = False,
) -> CRFSuiteAnnotator:
    algorithm = train_params.pop("algorithm")
    annotator = CRFSuiteAnnotator.for_training(
        mention_type, feature_extractor, mention_encoder
    )
    annotator.train(
        train_docs,
        tmp_model_path=tmp_model_path,
        algorithm=algorithm,
        train_params=train_params,
        verbose=verbose,
    )
    return annotator
