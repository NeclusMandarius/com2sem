#!/usr/bin/python3

# Author: Sebastian Schmidt
# Date: 05.07.2021

import logging
import re

from com2sem import Com2Sem
from com2sem.com import COM
import scipy
import numpy as np


def build_test_data(test_data_file):
    with open(test_data_file, encoding="utf-8") as file:
        test_data = map(lambda line: re.split(r"\s+", line[:-1].lower()), file.read().split("\n"))
    return list(test_data)


def cosine_dist(vec1, vec2, indices):
    return 1 - scipy.spatial.distance.cosine(vec1[indices], vec2[indices])


def test(corpus, min_samples_leaf, max_depth, criterion, test_data,
         space_definition="example-target-space/semantic_space.yaml",
         training_dataset="example-target-space/semantic_space.csv",
         filter_words=None):
    c2s = Com2Sem(
        corpus.get_orig_feature_names() if type(corpus) == COM else corpus[0],
        space_definition,
        classifier_opts=dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf[0], class_weight="balanced",
                             criterion=criterion[0], random_state=0),
        regressor_opts=dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf[1],
                            criterion=criterion[1], random_state=0)
    )
    if type(corpus) == COM:
        c2s.train(training_dataset, corpus.get_single_vector_ppmi)
    else:
        c2s.train(training_dataset, corpus[1], corpus[2])

    logging.info("Trained model! Min. samples leaf: {0}, max. depth: {1}, criterion: {2}", min_samples_leaf, max_depth,
                 criterion)

    reduced_vectors_datapoints = []
    nco_datapoints = []

    indices = c2s.get_used_orig_features()

    for line in test_data:
        word_a, word_b, sim = re.split(r"\s+", line[:-1].lower())
        if filter_words and word_a in filter_words and word_b in filter_words:
            continue

        sim = float(sim)
        vec1 = corpus.get_single_vector_ppmi(word_a) if type(corpus) == COM else corpus[2](corpus[1], word_a)
        if vec1 is False:
            continue
        vec2 = corpus.get_single_vector_ppmi(word_b) if type(corpus) == COM else corpus[2](corpus[1], word_b)
        if vec2 is False:
            continue

        reduced_vectors_datapoints.append((sim, cosine_dist(vec1, vec2, indices)))
        nco_datapoints.append((sim, c2s.normalized_concept_overlapping(vec1, vec2, False)))

    result = {}
    for dps, name in ((reduced_vectors_datapoints, "reduced_vectors"), (nco_datapoints, "com2sem")):
        for method, mname in ((scipy.stats.spearmanr, "spearman"), (scipy.stats.pearsonr, "pearson")):
            result[f"{name}-{mname}-correlation"] = method(
                [dp[0] for dp in dps], [dp[1] if not np.isnan(dp[1]) else 0 for dp in dps]
            )

    return result


def standard_test_suite(corpus, test_data,
                        space_definition="example-target-space/semantic_space.yaml",
                        training_dataset="example-target-space/semantic_space.csv"):
    filter_words = []
    with open(training_dataset, encoding="utf-8") as file:
        for line in file:
            filter_words.append(line.split(",", maxsplit=1)[0])

    results = []

    for max_depth in (2, 5, 10, 15, 20):
        for criterion in ("entropy", "gini"):
            results.append((
                [2, 2], max_depth, [criterion, "mse"],
                test(corpus, [2, 2], max_depth, [criterion, "mse"], test_data, space_definition, training_dataset)
            ))

    return results
