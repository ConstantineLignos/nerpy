import time
from abc import ABCMeta, abstractmethod
from os import PathLike
from typing import Iterable, Sequence, Union

from nerpy.document import Document, Mention
from nerpy.encoding import MentionEncoder
from nerpy.features import ExtractedFeatures, SentenceFeatureExtractor


class MentionAnnotator(metaclass=ABCMeta):
    @abstractmethod
    def mentions(self, doc: Document) -> Sequence[Mention]:
        raise NotImplementedError

    def add_mentions(self, doc: Document) -> Document:
        return doc.copy_with_mentions(self.mentions(doc))


class Trainable(metaclass=ABCMeta):
    @abstractmethod
    def train(self, docs: Iterable[Document], *args, **kwargs) -> None:
        raise NotImplementedError


class SequenceMentionAnnotator(MentionAnnotator, metaclass=ABCMeta):
    @property
    @abstractmethod
    def mention_encoder(self) -> MentionEncoder:
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_extractor(self) -> SentenceFeatureExtractor:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_path(cls, path: Union[str, PathLike]) -> "SequenceMentionAnnotator":
        raise NotImplementedError

    @abstractmethod
    def to_path(self, path: Union[str, PathLike]) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_bytes(cls, buf: bytes) -> "SequenceMentionAnnotator":
        raise NotImplementedError

    @abstractmethod
    def to_bytes(self) -> bytes:
        raise NotImplementedError
