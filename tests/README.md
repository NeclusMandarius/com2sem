# Com2Sem Tests

This folder contains the tests employed in the evaluation of my thesis.

## Datasets

Following datasets have been used for the specific test cases:

- Word Similarity: `WS-353-SIM`
- Word Clustering: `BM`
- Outlier Detection: `WikiSem-500`

All these datasets can be found here: https://github.com/vecto-ai/word-benchmarks

Furthermore, the test contained in `word_clustering_semcat.py` makes use of the
SEMCAT dataset: https://github.com/avaapm/SEMCATdataset2018

## Requirements
In addition to the requirements presented in the main README, some of the tests make use of
SciPy functionality (https://docs.scipy.org/doc/). Accordingly, this library needs to be installed in order to run
the tests.