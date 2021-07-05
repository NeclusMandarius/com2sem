#!/usr/bin/python3

# Author: Sebastian Schmidt
# Date: 05.07.2021
import csv
import logging
import re
from collections import defaultdict

from com2sem import Com2Sem
from com2sem.com import COM


def read_test_data(csv_file):
    with open(csv_file, encoding="utf-8") as file:
        test_data = list(csv.reader(file))

    return list(
        map(
            lambda y: [
                list(filter(lambda x: x.isalpha(), re.split(r"'(.*?)'", y[2]))),
                list(filter(lambda x: x.isalpha(), re.split(r"'(.*?)'", y[3])))
            ], test_data
        )
    )


def compactness_score(words, word_to_vectors, model):
    scores = defaultdict(lambda *x: 0)
    for i in words:
        for j in words:
            if i == j:
                continue
            scores[i] += model.normalized_concept_overlapping(word_to_vectors[i], word_to_vectors[j], False)
    for i in scores:
        scores[i] /= len(words) - 1
    return scores


def test(corpus, min_samples_leaf, max_depth, criterion, test_data,
         space_definition="example-target-space/semantic_space.yaml",
         training_dataset="example-target-space/semantic_space.csv"):
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

    max_number_meanings = 0
    outlier_detected = 0
    total_outliers = 0
    opp = 0
    act_lines = 0

    for outliers, words in test_data:
        if not outliers or not words:
            continue

        word_to_vectors = {}
        for word in (outliers + words):
            to_remove = False
            if type(corpus) == COM and word not in corpus.vocabulary:
                to_remove = True
            elif corpus[2](corpus[1], word) is False:
                to_remove = True
            if to_remove:
                if word in outliers:
                    outliers.remove(word)
                else:
                    words.remove(word)
                continue

            word_to_vectors[word] = list(
                c2s.predict(corpus.get_single_vector_ppmi(word) if type(corpus) == COM else corpus[2](corpus[1], word))
            )

            if len(word_to_vectors[word]) == 0:
                word_to_vectors.pop(word)
                if word in outliers:
                    outliers.remove(word)
                else:
                    words.remove(word)
                continue

            max_number_meanings = max(max_number_meanings, len(word_to_vectors[word]))

        if not outliers or len(words) <= 1:
            continue

        local_rate = 0
        for word in outliers:
            results = compactness_score([word] + words, word_to_vectors, c2s)
            index = 0
            for i in sorted(map(lambda x: tuple(reversed(x)), results.items()), reverse=True):
                if i[1] in outliers:
                    if index == len(results) - 1:
                        outlier_detected += 1
                        local_rate += 1
                    opp += index / len(words)
                    break
                index += 1
        logging.info("Line: {0} - {1}", outliers, words)
        logging.info("Accuracy: {0}\n", round(local_rate / len(outliers) * 100, 2))
        total_outliers += len(outliers)
        act_lines += 1

    return {
        "accuracy": outlier_detected / total_outliers * 100,
        "opp": opp / total_outliers * 100,
        "max-meanings": max_number_meanings,
        "num-lines-tested": act_lines
    }
