from pathlib import Path
from typing import Iterable, List, Optional, Sequence, TextIO, Tuple

from attr import attrib, attrs

from nerpy import Document, DocumentBuilder, MentionEncoder, Token
from nerpy.io import PathType

DOCSTART = "-DOCSTART-"


@attrs(frozen=True)
class CoNLLIngester:
    mention_encoder: MentionEncoder = attrib()
    ignore_comments: bool = attrib(default=False, kw_only=True)

    def ingest(self, source: TextIO, document_id_base: str) -> List[Document]:
        documents = []
        document_counter = 1
        builder = DocumentBuilder(document_id_base + "_" + str(document_counter))

        for source_sentence in self._parse_file(
            source, ignore_comments=self.ignore_comments
        ):
            sentence_tokens: List[Token] = []
            sentence_labels: List[str] = []

            if source_sentence[0].is_docstart:
                # We should only receive DOCSTART in a sentence by itself. This isn't a constraint on the document
                # lines; it's enforced by _parse_file which will break off sentence-initial DOCSTART. This assertion
                # is just checking the behavior of _parse_file and can't be hit by changing the data.
                assert (
                    len(source_sentence) == 1
                ), f"Received -DOCSTART- as part of a sentence at line {source_sentence[0].line_num}"

                # End current document and start a new one
                # We skip this if the builder is empty, which will happen for the very
                # first document in the corpus (as there is no previous document to end).
                if builder:
                    document = builder.build()
                    documents.append(document)
                    document_counter += 1
                    builder = DocumentBuilder(
                        document_id_base + "_" + str(document_counter)
                    )
                continue

            # Create mentions from tokens in sentence
            for idx, token in enumerate(source_sentence):
                new_token = Token.create(
                    token.text,
                    idx,
                    pos_tag=token.pos_tag,
                    chunk_tag=token.chunk_tag,
                    lemmas=token.lemmas,
                )
                sentence_tokens.append(new_token)
                sentence_labels.append(token.ne_tag)

            sentence = builder.create_sentence(sentence_tokens)

            mentions = self.mention_encoder.decode_mentions(sentence, sentence_labels)
            builder.add_mentions(mentions)

        document = builder.build()
        documents.append(document)

        return documents

    @classmethod
    def _parse_file(
        cls, input_file: TextIO, *, ignore_comments: bool = False
    ) -> Iterable[Tuple["_CoNLLToken", ...]]:
        sentence: list = []
        line_num = 0
        for line in input_file:
            line_num += 1
            line = line.strip()

            if ignore_comments and line.startswith("#"):
                continue

            if not line:
                # Clear out sentence if there's anything in it
                if sentence:
                    yield tuple(sentence)
                    sentence = []
                # Always skip empty lines
                continue

            token = cls._CoNLLToken.from_line(line, line_num)
            # Skip document starts, but ensure sentence is empty when we reach them
            if token.is_docstart:
                if sentence:
                    raise ValueError(
                        f"Encountered DOCSTART at line {line_num} while still in sentence"
                    )
                else:
                    # Yield it by itself
                    yield (token,)
            else:
                sentence.append(token)

        # Finish the last sentence if needed
        if sentence:
            yield tuple(sentence)

    @attrs(frozen=True)
    class _CoNLLToken:
        text: str = attrib()
        pos_tag: Optional[str] = attrib()
        lemmas: Optional[Tuple[str, ...]] = attrib()
        chunk_tag: Optional[str] = attrib()
        ne_tag: str = attrib()
        is_docstart: bool = attrib()
        line_num: int = attrib()

        @classmethod
        def from_line(cls, line: str, line_num: int) -> "CoNLLIngester._CoNLLToken":
            splits = line.split()
            text = splits[0]
            ne_tag = splits[-1]

            if len(splits) == 5:
                # Assume has lemmas like 2002 German data
                lemmas = tuple(splits[1].split("|"))
                pos_tag = splits[2]
                chunk_tag = splits[3]
            else:
                lemmas = None
                # Other tags will be POS if available, then chunk if available
                pos_tag = splits[1] if len(splits) > 2 else None
                chunk_tag = splits[2] if len(splits) > 3 else None

            is_docstart = text == DOCSTART
            return cls(text, pos_tag, lemmas, chunk_tag, ne_tag, is_docstart, line_num)


def read_conll(
    path: PathType,
    mention_encoder: MentionEncoder,
    *,
    document_id_base: Optional[str] = None,
    ignore_comments: bool = False,
) -> List[Document]:
    ingester = CoNLLIngester(mention_encoder, ignore_comments=ignore_comments)

    # Create document_id_base from filename if needed
    if document_id_base is None:
        document_id_base = Path(path).name

    with open(path, encoding="utf8") as file:
        return ingester.ingest(file, document_id_base)


def write_conll(
    docs: Sequence[Document],
    output_path: PathType,
    mention_encoder: MentionEncoder,
    lang: Optional[str] = None,
) -> None:
    # TODO: Check that this can round-trip Spanish data correctly
    # Figure out how many fields to output by seeing how many of the CoNLL
    # fields are present
    sample_tok = docs[0][0][0]
    has_lemmas = sample_tok.lemmas is not None
    has_pos = sample_tok.pos_tag is not None
    has_chunk = sample_tok.chunk_tag is not None
    # Add as many additional fields as needed by counting the number of Trues
    # The additional fields are -X- normally, but -DOCSTART- for the Dutch data
    docstart_fields = [DOCSTART]
    for _ in range(sum((has_lemmas, has_pos, has_chunk))):
        docstart_fields.append("-X-" if lang != "ned" else DOCSTART)
    docstart_fields.append("O")
    docstart_line = " ".join(docstart_fields)

    with open(output_path, "w", encoding="utf8") as output_file:
        for doc in docs:
            # Don't output docstart if there's only one document
            if len(docs) > 1:
                print(docstart_line, file=output_file)
                # For Dutch data, no blank line after docstart
                if lang != "ned":
                    print(file=output_file)
            for sentence, mentions in doc.sentences_with_mentions():
                tokens = sentence.tokens
                try:
                    labels = mention_encoder.encode_mentions(sentence, mentions)
                except ValueError as e:
                    raise ValueError(
                        f"Error writing document {doc.id} sentence {sentence.index} "
                        f"with mentions: {mentions}",
                    ) from e
                for token, label in zip(tokens, labels):
                    line_fields = [token.text]
                    if has_lemmas:
                        line_fields.append("|".join(token.lemmas))
                    if has_pos:
                        line_fields.append(token.pos_tag)
                    if has_chunk:
                        line_fields.append(token.chunk_tag)
                    line_fields.append(label)
                    line = " ".join(line_fields)
                    print(line, file=output_file)
                print(file=output_file)
