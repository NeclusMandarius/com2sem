#!/usr/bin/python3

# Author: Sebastian Schmidt
# Date: 05.07.2021
import csv
import logging
from collections import defaultdict

from com2sem import Com2Sem
from com2sem.com import COM

# Cluster/concept mapping for the Battig dataset
CLUSTER_TO_CONCEPT = {
    "precious stone": "substance",
    "unit of time": "quantity",
    "relative": "human",
    "unit of distance": "quantity",
    "metal": "substance",
    "type of reading material": "info",
    "military title": "human",
    "four-footed animal": "animal",
    "kind of cloth": "clothes",
    "color": "quality",
    "kitchen utensil": "instrument",
    "bldg for religious servic": "building",
    "part of speech": "language",
    "article of furniture": "instrument",
    "part of the human body": "animate",
    "fruit": "plant",
    "weapon": "instrument",
    "elective office": "human",
    "type of human dwelling": "building",
    "alcoholic beverage": "substance",
    "country": "constructed",
    "crime": "situation",
    "carpenters tool": "instrument",
    "member of the clergy": "human",
    "substance to flavor food": "substance",
    "type of fuel": "substance",
    "occupation or profession": "human",
    "natural earth formation": "geographical",
    "sport": "discipline",
    "weather phenomenon": "dynamic situation",
    "article of clothing": "clothes",
    "part of a building": "building",
    "chemical element": "substance",
    "musical instrument": "instrument",
    "kind of money": "quantity",
    "type of music": "discipline",
    "bird": "animal",
    "nonalcoholic beverage": "substance",
    "type of vehicle": "inanimate",
    "science": "discipline",
    "toy": "instrument",
    "type of dance": "discipline",
    "vegetable": "plant",
    "type of footgear": "clothes",
    "insect": "animal",
    "girls first name": "human",
    "males first name": "human",
    "flower": "plant",
    "disease": "discipline",
    "tree": "plant",
    "type of ship": "inanimate",
    "fish": "animal",
    "snake": "animal",
    "city": "constructed",
    "state": "constructed",
    "college or university": "constructed"
}


def build_test_data(csv_file):
    with open(csv_file, encoding="utf-8") as file:
        test_data = list(csv.reader(file))
    test_clusters = defaultdict(list)
    for line in test_data:
        test_clusters[line[1]].append(line[2])
    return test_clusters


def test(corpus, min_samples_leaf, max_depth, criterion, test_clusters,
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
    average_purity = 0  # Summed purities over all clusters
    total_inliers = 0  # Number of all words that are annotated for any cluster (may be taken into account more than
    # once, if occurring for more than one cluster) -- words not contained in the corpus are ignored
    inlier_detection = 0  # Number of words being predicted correctly for a cluster (may be taken into account more than
    # once, if occurring for more than one cluster)
    total_clusters = 0  # Number of processed clusters (clusters not suitably represented in the corpus are ignored)
    average_length = 0  # Sum of the number of meanings for all words correctly predicted for a cluster

    for c in test_clusters:
        word_to_vectors = {}
        for word in test_clusters[c]:
            if type(corpus) == COM and word not in corpus.vocabulary:
                continue
            elif corpus[2](corpus[1], word) is False:
                continue

            word_to_vectors[word] = list(
                c2s.predict(corpus.get_single_vector_ppmi(word) if type(corpus) == COM else corpus[2](corpus[1], word))
            )

            if len(word_to_vectors[word]) == 0:
                word_to_vectors.pop(word)
                continue
            max_number_meanings = max(max_number_meanings, len(word_to_vectors[word]))
        if len(word_to_vectors) < 2:
            logging.warning("Skipping \"{0}\" since it contains less than two words.", c)
            continue

        res = []
        for w in word_to_vectors:
            yes = -1
            for e in word_to_vectors[w]:
                yes = max(yes, e[c2s.configuration.features.index(CLUSTER_TO_CONCEPT[c])])

            if yes > 0.5:
                yes = 1
                average_length += len(word_to_vectors[w])

            res.append(yes)

        purity = res.count(1) / len(res)
        logging.info("Purity for cluster \"{0}\": {1}", c, round(purity * 100, 2))

        average_purity += purity
        total_clusters += 1
        inlier_detection += res.count(1)
        total_inliers += len(res)

    return {
        "avg-purity": round(average_purity / total_clusters * 100, 2),
        "accuracy": round(inlier_detection / total_inliers * 100, 2),
        "max-meanings": max_number_meanings,
        "avg-meanings": round(average_length / inlier_detection, 2),
        "total-words": total_inliers
    }
