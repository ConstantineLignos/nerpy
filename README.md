# NERPy

## Scripts

All scripts should be run from the root of the repository, i.e. `python scripts/ingest_conll.py`.

See `demo.sh` for a demonstration of how to train and test a model using these scripts.

### Ingest

* `ingest_conll.py`: Ingest CoNLL format files into the NERPy pickle format
* `ingest_ontonotes.py`: Ingest OntoNotes .name format files into the NERPy pickle format
* `convert_embedding.py`: Convert a word embedding from the .vec format into a SQLite database for use in NERPy

### Training and testing

All training and test scripts require that the input be ingested.

* `train.py`: Train a model
* `test.py`: Test a model
* `train_test.py`: Train and test a model


### Scoring

* `score_conll.py`: Score CoNLL format files 


### Debugging

* `demo_embeddings.py`: Test the embeddings module

## Development

Set up for development as follows:
```
pip install -e .[dev]
```

Before committing changes, run `tests/pre_commit.sh` to run autoformatting, static analysis, and tests.
