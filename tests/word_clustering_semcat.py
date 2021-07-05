#!/usr/bin/python3

# Author: Sebastian Schmidt
# Date: 05.07.2021
import csv
import logging
import os
from collections import defaultdict

from com2sem import Com2Sem
from com2sem import conf
from com2sem.com import COM

N_train = 20


def prepare_categories(categories_path=None):
    """
    Collects the filenames of all categories that are used by the semcat evaluation (thus specified in the
    tests/c2s_eval_config.yaml).

    :param categories_path: path to SEMCAT category folder
    :return: tuple of the Com2Sem test configuration and a dictionary mapping the category name to its file
    """
    assert categories_path is not None, "You have to download and specify the SEMCAT dataset first!"

    config = conf.Configuration("tests/c2s_eval_config.yaml")
    config.build()

    category_to_file = dict()
    for fp in os.listdir(categories_path):
        category_to_file[fp.split("-", maxsplit=1)[0].replace("_", " ")] = os.path.join(categories_path, fp)

    return config, category_to_file


def build_corpus(corpus_file):
    corpus = COM(corpus_file, limit=100_000_000, window_mapping=3)
    corpus.build(end_index=0, preprocessor=lambda x: x.lower(), filter=(corpus._bigram_measure, 0))
    return corpus


def prepare_training_data(config, category_to_file, csv_filename="tests/c2s_eval_corpus.csv"):
    """
    Creates a CSV file containing the training data and collects the annotated embeddings for all other words.

    :param config: Com2Sem test configuration
    :param category_to_file: mapping of SEMCAT category names to their definition files
    :param csv_filename: file where to save the training data
    :return: tuple of the training data filename and a mapping of non-training words to their correct embeddings
    """
    words = []
    word_dict = defaultdict(list)
    for e in config.features:
        if e in category_to_file:
            with open(category_to_file[e]) as file:
                categ_words = []
                index = 0

                while True:
                    word = file.readline().strip()
                    if not word:
                        break

                    vec = [-1] * len(config.features)
                    vec[config.features.index(e)] = 1
                    if index < N_train:
                        words.append([word] + vec)
                        index += 1
                        continue
                    word_dict[word].append(vec)
                    categ_words.append(word)
                    index += 1
    with open(csv_filename, "w", newline="") as file:
        wr = csv.writer(file)
        for w in words:
            wr.writerow(w)

    return csv_filename, word_dict


def test(corpus, config, min_samples_leaf, max_depth, criterion, word_dict,
         space_definition="tests/c2s_eval_config.yaml",
         training_dataset="tests/c2s_eval_corpus.csv"):
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

    whole_target_matrix = {}
    target_words = defaultdict(list)

    for w in word_dict:
        try:
            vecs = c2s.predict(corpus.get_single_vector_ppmi(w))
        except:
            continue
        whole_target_matrix[w] = vecs
        for e in vecs:
            ft = config.features[e.argmax()]
            target_words[ft].append(w)

    target_words_labels = defaultdict(list)
    for i in word_dict:
        for e in word_dict[i]:
            nm = config.features[e.index(1)]
            target_words_labels[nm].append(i)

    num_all = 0
    num_true_positives = 0
    single_results = {}
    for e in sorted(target_words_labels.keys()):
        s1 = set(target_words_labels[e])
        s2 = set(target_words[e]) if e in target_words else set()
        s3 = s1.intersection(s2)

        single_results[e] = {
            "accuracy": round(len(s3) / len(s1) * 100, 2),
            "num-false-positives": len(s2.difference(s1)),
            "num-annotated-positives": len(s1)
        }

        num_all += len(s1)
        num_true_positives += len(s3)

    return {
        "accuracy": round(num_true_positives / num_all * 100, 2),
        "single-results": single_results
    }


def standard_test_suite(corpus, categories_path,
                        space_definition="example-target-space/semantic_space.yaml",
                        training_dataset="example-target-space/semantic_space.csv"):
    config, category_to_file = prepare_categories(categories_path)
    csv_name, word_dict = prepare_training_data(config, category_to_file)
    results = []

    for max_depth in (1, 15):
        for criterion in ("entropy", "gini"):
            results.append((
                [2, 2], max_depth, [criterion, "mse"],
                test(
                    corpus, config, [2, 2], max_depth, [criterion, "mse"], word_dict, space_definition, training_dataset
                )
            ))

    return results
