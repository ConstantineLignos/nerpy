import json
import pickle
from os import PathLike
from pathlib import Path
from typing import Dict, List, Union

from nerpy.document import Document

# Union[str, Path] isn't enough to appease PyCharm's type checker, so adding Path here
# avoids warnings.
PathType = Union[str, Path, PathLike]


def load_pickled_documents(path: PathType) -> List[Document]:
    with open(path, "rb") as file:
        return pickle.load(file)


def load_pickled_document(path: PathType) -> Document:
    with open(path, "rb") as file:
        return pickle.load(file)


def pickle_documents(docs: List[Document], path: PathType) -> None:
    with open(path, "wb") as file:
        return pickle.dump(docs, file, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_document(doc: Document, path: PathType) -> None:
    with open(path, "wb") as file:
        return pickle.dump(doc, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(path: PathType) -> Dict:
    with open(path, encoding="utf8") as file:
        return json.load(file)
