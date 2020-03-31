#! /usr/bin/env bash
set -euxo pipefail

# Entity encoding of the CoNLL files (BIO, etc.)
ENCODING=BIO

# Path to the training data, in CoNLL format
TRAIN=data/conll2003/en/train.txt

# Path to the test data, in CoNLL format
TEST=data/conll2003/en/test.txt

# Output directory
OUTPUT=output
mkdir -p $OUTPUT

# Ingest data
python scripts/ingest_conll.py $TRAIN $ENCODING $OUTPUT/train.nerpy
python scripts/ingest_conll.py $TEST $ENCODING $OUTPUT/test.nerpy

# Train a simple model
python scripts/train.py $OUTPUT/train.nerpy $OUTPUT/conll.model params/training/crfsuite_ap_20iter.json params/features/baseline.json $ENCODING

# Test the model
python scripts/test.py $OUTPUT/conll.model $OUTPUT/test.nerpy params/features/baseline.json $ENCODING $OUTPUT/test_predictions.nerpy
