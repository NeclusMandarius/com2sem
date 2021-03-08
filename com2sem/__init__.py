#!/usr/bin/python3

# Author: Sebastian Schmidt
# Date: 14.02.2021

import csv
import os
from typing import Dict, List, Union

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from nltk.tokenize import word_tokenize
except ImportError:
    word_tokenize = None

from conf import Configuration


class Com2Sem:
    """
    The Com2Sem model provides the package's main functionality: it can be trained on a co-occurrence
    matrix (or another object from which co-occurrence vectors can be constructed) to find decision trees
    recognizing specific meanings dependent on a number of single context tokens. The target space structure,
    defining e.g. the number of decision trees by its dimensionality, has to be defined using a target space
    configuration file.

    The model is not only able to predict meanings to a given vector afterwards, but also to display or export the
    trained decision trees, and even to find justification for specific predictions.

    For an implementation of decision trees, this package bases on the Scikit-learn package. By default, they
    are configured as follows (only parameters deviating from the sklearn default values):

    Classifier trees:
     - max_depth = 15
     - min_samples_leaf = 2
     - class_weight = "balanced"

    Regressor trees:
     - max_depth = 15

    In the evaluation it appeared, as if the "gini" criterium (default) generally performed better for higher maximum
    depths whereas the "entropy" criterion consistenly achieved higher accuracies for very low depths. (Both criteria
    only apply to classifier trees)

    The target space is very important for the model's performance; yet, no completely appropriate has been found so
    far. It might be preferable to first apply it to very specific tasks, or to use an automated approach to build a
    complex target-space performing reasonably well.
    """

    orig_feature_names: List[str] = None
    configuration: Configuration = None

    trees: List[Union[None, DecisionTreeClassifier, DecisionTreeRegressor]] = None

    classifier_opts = dict(max_depth=15, min_samples_leaf=2, class_weight="balanced", criterion="entropy")
    regressor_opts = dict(max_depth=15)

    def __init__(self, orig_feature_names: List[str], configuration_file: str, classifier_opts=None,
                 regressor_opts=None):
        """
        Initializes a new Com2Sem model.

        :param orig_feature_names: list of words used as columns for the co-occurrence vectors, the order has to match
               the co-occurrence vectors
        :param configuration_file: path of a target space configuration file, refer to `conf.py` for more information
        :param classifier_opts: optional dictionary to customize the configuration for classifier trees
        :param regressor_opts: optional dictionary to customize the configuration for regressor trees

        :exception ValueError: thrown if the target space configuration could not be loaded
        """
        if regressor_opts is None:
            regressor_opts = {}
        if classifier_opts is None:
            classifier_opts = {}

        self.orig_feature_names = orig_feature_names
        self.configuration = Configuration(configuration_file)
        try:
            self.configuration.build()
        except:
            raise ValueError("Error while parsing the target space configuration!")

        self.trees = list(map(lambda x: None, self.configuration.features))

        self.classifier_opts = Com2Sem.classifier_opts.copy()
        self.regressor_opts = Com2Sem.regressor_opts.copy()
        self.classifier_opts.update(classifier_opts)
        self.regressor_opts.update(regressor_opts)

    def _init_training_data(self, training_file, co_occurence_lookup, co_occurence_matrix, dtype):
        hcvecsl = [], []

        with open(training_file, encoding="utf-8") as file:
            hcvecs = csv.reader(file)
            line = 0
            for i in hcvecs:
                assert len(i) == len(self.configuration.features) + 1
                line += 1
                if i[0].startswith("#") or (line == 1 and i[1:] == self.configuration.features):
                    continue
                if co_occurence_matrix is None:
                    lookup = co_occurence_lookup(i[0])
                else:
                    lookup = co_occurence_lookup(i[0], co_occurence_matrix)
                if lookup is False:
                    continue
                hcvecsl[0].append(lookup.astype(dtype))
                hcvecsl[1].append(list(map(float, i[1:])))

        return np.array(hcvecsl[0], dtype=dtype), np.array(hcvecsl[1], dtype=np.float16)

    def train(self, training_file: str, co_occurrence_lookup, co_occurrence_matrix=None, balancing=True,
              dtype=np.ushort):
        """
        This method can be used to train the model by supplying training data contained in a CSV file. It is furthermore
        necessary to provide an object representing the co-occurrence matrix and a suitable function, constructing
        co-occurrence vectors from it for any given word. For training the decision trees, as to this point
        two dense matrices are constructed for the training data, possibly leading to high memory consumption;
        this might be subject to change in the future. The second matrix can be avoided when setting the `balancing`
        parameter to False (refer to the parameter description).

        :param training_file: file path to a CSV file containing training data, see the README's training data section
        :param co_occurrence_lookup: a function or lambda expression returning a word's co-occurrence vector. Expects
                                     the word as string and optionally an object representing the co-occurrence matrix.
                                     The last parameter is supplied only if `co_occurrence_matrix` has been defined.
                                     IMPORTANT: if for a word no or no reliable data has been collected, the function
                                     should return False so that the respective piece of training data can be ignored.
        :param co_occurrence_matrix: an arbitrary object passed to the lookup function as second argument if not None
        :param balancing: if True (default) adds enough pure zero-vectors classified as -1 to the training data of each
                          *classifier* tree to meet the `min_samples_leaf` criterion. This should prevent the tree from
                          predicting the positive class by default -- it is not completely reliable, though. Turning
                          off the parameter can save memory as then no copy of the original co-occurrence matrix has to
                          be created (nevertheless copies of sub-matrices for trees in deeper levels, if there are any).
        :param dtype: defines the NumPy data type of the co-occurrence matrices; numpy.ushort by default for scaled up
                      PPMI matrices as constructed by the `com` module. Using other datatypes could increase the memory
                      usage drastically. The training data's labels are saved in a np.float16 matrix.
        :return: void
        """
        training_data, training_labels = self._init_training_data(training_file, co_occurrence_lookup,
                                                                  co_occurrence_matrix, dtype)

        self._train(self.configuration.hierarchy, training_data, training_labels,
                    np.fromiter(range(training_labels.shape[0]), np.uint), balancing, dtype)

    def _train(self, level, training_data, training_labels, indices, balancing, dtype):
        uses = level['_uses'] if '_uses' in level else {}
        new_indices = set(indices)
        for e in uses:
            f_index = self.configuration.features.index(e)
            if uses[e] is None or not uses[e][1]:
                self.trees[f_index] = DecisionTreeRegressor(**self.regressor_opts) \
                    .fit(training_data[indices], training_labels[indices].T[f_index])
            else:
                # Sort out training data violating restrictions of regressor feature values
                for i in range(len(training_labels)):
                    if training_labels[i][f_index] != uses[e][0]:
                        new_indices.difference_update((i,))
        indices = np.fromiter(new_indices, np.uint)

        if len(indices) == 0:
            return

        min_samples_leaf = self.classifier_opts["min_samples_leaf"] \
            if "min_samples_leaf" in self.classifier_opts else 1

        for i in level:
            if i.startswith("_"):
                continue
            f_index = self.configuration.features.index(i)
            clf_opts = self.classifier_opts.copy()
            if "class_weight" in clf_opts and type(clf_opts["class_weight"]) == dict:
                # Sklearn throws an exception if class_weights are defined per-class and not all of these classes are
                # present, so in this case we remove the respective items
                for e in {-1, 1}.difference(set(training_labels[indices].T[f_index])):
                    if e in clf_opts["class_weight"]:
                        clf_opts["class_weight"].pop(e)
            self._tree_train(
                f_index, training_data, training_labels, indices, min_samples_leaf, clf_opts, balancing, dtype
            )
            if level[i] is not None:
                to_observe = np.nonzero(training_labels.T[f_index] == 1)[0]
                self._train(level[i], training_data, training_labels, to_observe, balancing, dtype)

    def _tree_train(self, f_index, training_data, training_labels, indices, min_samples_leaf, clf_opts, balancing,
                    dtype):
        if balancing:
            tree_train = (
                np.concatenate(
                    (training_data[indices], np.zeros((min_samples_leaf, training_data.shape[1]), dtype))
                ),
                np.concatenate(
                    (training_labels[indices].T[f_index], np.zeros((min_samples_leaf,), np.float16) - 1)
                )
            )
        elif indices.shape[0] == training_data.shape[0]:
            # Avoid constructing a new matrix if we can re-use the old one
            tree_train = (training_data, training_labels.T[f_index])
        else:
            tree_train = (training_data[indices], training_labels[indices].T[f_index])
        self.trees[f_index] = DecisionTreeClassifier(**clf_opts) \
            .fit(*tree_train)

    def predict(self, vector: np.array, threshold_function=None, regressor_wrapper=None, **kwargs):
        """
        Used to predict embeddings from a co-occurrence vectors. In contrast to many embedding models, this method
        is able to find several meaning contexts from one vector, by following all possible paths until a tree does not
        predict a concept as positive with enough certainty. A custom threshold function can be applied to change the
        decision behavior of the model (e.g. taking into account knowledge of other sources).

        :param vector: a co-occurrence vector for the word that should be embedded in the target space
        :param threshold_function: a function or lambda expression that is called for every classifier tree used in the
                                   prediction process to determine whether its assigned concept is accepted or not. It
                                   should expect at least two arguments: the probability for positive classification
                                   output by the respective tree and the predicetd feature's name. Furthermore, this
                                   method's **kwargs arguments are passed on to the function.
        :param regressor_wrapper: a function or lambda expression that is called for every regressor tree used in the
                                  prediction process to possibly change its predicted value. It should expect at least
                                  two arguments: the regression tree's output and the predicetd feature's name.
                                  Furthermore, this method's **kwargs arguments are passed on to the function.
        :param kwargs: all **kwargs are directly passed onto the threshold_function and the regressor_wrapper and may
                       e.g. contain the actual predicted word as string.
        :return: a two-dimensional numpy.array containing as rows all predicted embeddings
        """
        assert vector is not False, "The given vector is False!"

        result = np.zeros((len(self.configuration.features),)) - 1

        if threshold_function is None:
            threshold_function = lambda prob, fname, **kws: prob > 0.5

        if regressor_wrapper is None:
            regressor_wrapper = lambda value, fname, **kws: value

        return self._predict(self.configuration.hierarchy, vector, result, threshold_function, regressor_wrapper,
                             **kwargs)

    @staticmethod
    def _check_regression_domain(value):
        if 0 <= value <= 1:
            return value
        return -1

    def _predict(self, level, vector, result, decision_function, regressor_wrapper, **kwargs):
        uses = level['_uses'] if '_uses' in level else {}

        for e in uses:
            f_index = self.configuration.features.index(e)
            if (uses[e] is None or uses[e][1] is False) and self.trees[f_index] is not None:
                pred_val = self._check_regression_domain(self.trees[f_index].predict([vector])[0])
                result[f_index] = self._check_regression_domain(regressor_wrapper(pred_val, e))
            else:
                result[f_index] = uses[e][0]

        sub = []

        for i in level:
            if i.startswith("_"):
                continue
            f_index = self.configuration.features.index(i)
            if self.trees[f_index] is None or max(self.trees[f_index].classes_) == -1:
                sub.append((0, i))
            elif self.trees[f_index].n_classes_ == 1 and self.trees[f_index].classes_[0] == 1:
                sub.append((1, i))
            else:
                sub.append((self.trees[f_index].predict_proba([vector])[0][1], i))
        sub.sort(reverse=True)

        results = []
        for e in sub:
            if decision_function(*e, **kwargs):
                res_temp = result.copy()
                res_temp[self.configuration.features.index(e[1])] = 1
                if level[e[1]] is None:
                    results.append(res_temp)
                else:
                    other_results = self._predict(level[e[1]], vector, res_temp, decision_function, regressor_wrapper,
                                                  **kwargs)
                    results.extend(other_results)
                    if len(other_results) == 0:
                        results.append(res_temp)

        return np.array(results)

    def get_tree(self, feature: str) -> Union[DecisionTreeClassifier, DecisionTreeRegressor, None]:
        """
        Retrieves the decision tree for a specific feature of the target space.

        :param feature: the name of one of the target space's features
        :return: the decision tree assigned the respective feature or None if it has not been trained yet

        :exception AssertionError: if the given feature does not exist in the target space
        """
        assert feature in self.configuration.features, f"Feature \"{feature}\" is not defined by the target space!"
        return self.trees[self.configuration.features.index(feature)]

    def show_tree(self, feature: str, **kwargs):
        """
        Wrapper for `sklearn.tree.plot_tree` for the specified feature's decision tree. The co-occurrence vector's
        indices are automatically replaced by the context word they represent in the resulting decision diagram.

        :param feature: the name of one of the target space's features
        :param kwargs: The **kwargs are directly passed onto the `plot_tree` function
        :return: the result of `sklearn.tree.plot_tree` if the feature's tree has already been trained, else False

        :exception AssertionError: if the given feature does not exist in the target space
        """
        tree = self.get_tree(feature)
        if tree is None:
            return False
        return sklearn.tree.plot_tree(tree, fontsize=5, feature_names=self.orig_feature_names, **kwargs)

    def export_tree(self, feature: str, file_path: str):
        """
        Wrapper for `sklearn.tree.export_graphviz` for the specified feature's decision tree. The co-occurrence vector's
        indices are automatically replaced by the context word they represent in the resulting decision diagram.

        :param feature: the name of one of the target space's features
        :param file_path: file path of where the exported DOT file should be saved
        :return: False if the requested tree is not trained yet, else None

        :exception AssertionError: if the given feature does not exist in the target space
        """
        tree = self.get_tree(feature)
        if tree is None:
            return False
        with open(file_path, "w") as file:
            sklearn.tree.export_graphviz(tree, feature_names=self.orig_feature_names, out_file=file)

    def export_model(self, path, model_name):
        """
        Uses `export_tree` for all features of the model, as long as they already possess a trained decision tree.

        :param path: folder path of where the exported DOT files should be saved
        :param model_name: base name used for every exported DOT file, suffixed by an underscore, the respective
                           feature name and the file ending ".dot"
        :return: void
        """
        for feature_name in self.configuration.features:
            self.export_tree(feature_name, os.path.join(path, model_name + "_" + feature_name + ".dot"))

    def decision_path(self, vector, feature) -> List[Dict]:
        """
        Outputs a decision path of the specified feature's decision tree for the given co-occurrence vector.

        :param vector: a co-occurrence vector for the word of which justification is wanted
        :param feature: the name of one of the target space's features; alternatively, its index can be given directly
        :return: a top-down ordered list containing a dictionary per encountered node with the following properties:
          - node: the node's id inside the tree
          - feature: context word the split observes
          - value: value contained in the co-occurrence vector for the observed feature
          - threshold: threshold used by the split
          - inequality: either "<=" or ">" depending on whether `threshold` is higher or lower than `value`
        """
        clf = self.trees[self.configuration.features.index(feature)] if type(feature) == str else self.trees[feature]
        sample = vector

        node_indicator = clf.decision_path([sample])
        leaf_id = clf.apply([sample])
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        sample_id = 0
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]

        result = []
        for node_id in node_index:
            if leaf_id[sample_id] == node_id:
                continue
            if sample[feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            result.append(dict(
                node=node_id,
                feature=self.orig_feature_names[feature[node_id]],
                value=sample[feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id]
            ))
        return result

    def justify(self, word, vec, features, corpus, language="english", inner_window_radius=2, outer_window_radius=3):
        """
        Given a word, its co-occurrence vector, one or more target space features and a corpus, tries to find word
        contexts which lead the decision trees to their classification results. The corpus can either be tokenized
        already (i.e. an iterable containing strings) or the actual corpus as string, in the case of which the
        function `nltk.tokenize.word_tokenize` is used for the specified `language` ("english" by default). For this,
        the NLTK Python package has to be installed, otherwise an exception is raised.

        Essentially, the method calculates all decision paths for the specified features, ignores negatively
        discriminating splits as the *not*-co-occurrence of a word can hardly be shown, and then iterates to the corpus
        to find contexts in which the given word appears together with at least one positively discriminating context
        word in the decision paths. The context size used for co-occurrence detection is specified by the
        `inner_window_radius` parameter, the `outer_window_radius` determines how many parts of the respective contexts
        are provided for the result (even if one only wants to search for directly adjacent context words, it can be
        helpful to know more of the context).

        The method acts as an iterator and yields results as they are found.

        :param word: the word of which a co-occurrence vector is given
        :param vec: the given word's co-occurrence vector
        :param features: the names or indices of target space features, decisions of which should be justified
        :param corpus: an iterable containing the already tokenized corpus or a string tokenized using the NLTK package
        :param language: language specified for the `nltk.tokenize.word_tokenize` function ("english" by default), only
                         relevant if the corpus is given as a string
        :param inner_window_radius: radius around the center word to be observed when searching for context words
        :param outer_window_radius: radius around the center word to be provided whenever a match is found
        :return: an iterator providing a tuple with
          0. all tokens in a radius of `outer_window_radius` around the center word as a list
          1. a list with tuples ("[context word]", "[feature name]") where "[feature name]" denotes the feature tree
             observing "[context word]" for a split. Note that the same context word can theoretically occur several
             times in the list if it is selected for several feature trees.

        :exception AssertionError: if a) the corpus is given as string and NLTK is not installed
                                   or b) the corpus is not iterable and thus can not be converted to a list
        """
        assert type(corpus) != str or word_tokenize is not None, "If a text corpus is given, the NLTK python package " \
                                                                 "must be installed so its `word_tokenize` function " \
                                                                 "can be used!"
        assert hasattr(corpus, "__iter__"), "An already tokenized corpus must be iterable!"

        if type(features) != list:
            features = [features]

        look_for = set()
        for feature in features:
            path = self.decision_path(vec, feature)
            look_for.update(
                list(map(lambda x: (x["feature"], feature), filter(lambda x: x["inequality"] == ">", path)))
            )

        if type(corpus) == str:
            tokens = word_tokenize(corpus, language)
        elif type(corpus) == list:
            tokens = corpus
        else:
            tokens = list(corpus)
        clen = len(tokens)

        ix = 0

        for t in tokens:
            if t == word:
                inner_context = tokens[max(ix - inner_window_radius, 0):min(ix + inner_window_radius, clen)]
                match = [i for i in look_for if i[0] in inner_context]
                if len(match) > 0:
                    outer_context = tokens[max(ix - outer_window_radius, 0):min(ix + outer_window_radius, clen)]
                    yield outer_context, match
            ix += 1

    def _C(self, *vecs):
        return [
            i for i in range(len(self.configuration.features))
            if type(self.trees[i]) == DecisionTreeClassifier and min(j[i] for j in vecs) == 1
        ]

    def normalized_concept_overlapping(self, vec_a, vec_b, show_level=True):
        """
        Method to evaluate the similarity of two embeddings vec_a and vec_b, similar to the Jaccard index:
        the number of commonly selected classifier features for both embeddings is divided by their maximum individual
        number of selected classifier features. Additionally, the difference of regressor features not being -1 in both
        features is applied as a negative modificator, but normalized by the maximum individual number of selected
        classifier features as well. Therefore, the deeper the classifier feature level the less influence do regressor
        features show on the result. If no classifier features are defined for any of the two embeddings, the score is
        always 0.

        The score has a domain of [0, 1]. Note that even for a score of 1, regressor values can still differ if one of
        them is set to -1.

        vec_a and vec_b can be iterables containing several embeddings as well, in the case of which the maximum score
        of all combinations is returned.

        If `show_level` is True (default), along with the score the deep-most common classifier feature is returned.

        :param vec_a: embedding (or iterable containing embeddings) in the target space
        :param vec_b: embedding (or iterable containing embeddings) in the target space
        :param show_level: if True (default) return the deep-most common classifier feature as well
        :return: if `show_level` is True a tuple (similarity: float, feature_name: str), else only the similarity
        """
        max_sim = 0.
        max_level = None

        if not hasattr(vec_a, "__iter__"):
            vec_a = [vec_a]
        if not hasattr(vec_b, "__iter__"):
            vec_b = [vec_b]

        for i in vec_a:
            for j in vec_b:
                level = self.configuration.hierarchy
                level_name = None
                stop = False
                sim_binary = 0
                sim_compatible = 0
                num_compatible = 0
                while not stop and level is not None:
                    uses = level["_uses"] if "_uses" in level else {}
                    for e in uses:
                        f_index = self.configuration.features.index(e)
                        if min(i[f_index], j[f_index]) >= 0:
                            sim_compatible += abs(i[f_index] - j[f_index])
                            num_compatible += 1
                    for e in level:
                        if e.startswith("_"):
                            continue
                        f_index = self.configuration.features.index(e)
                        if min(i[f_index], j[f_index]) == 1:
                            sim_binary += 1
                            level = level[e]
                            level_name = e
                            break
                    else:
                        stop = True
                sim = (sim_binary - (sim_compatible / max(num_compatible, 1)))
                sim /= max(len(self._C(i)), len(self._C(j)), 1)
                if sim > max_sim:
                    max_level = level_name
                max_sim = max(max_sim, sim, 0)
        return (max_sim, max_level) if show_level else max_sim

    def render_embedding(self, embedding, indent='    '):
        """
        Return a textual representation of the given embedding.
        Every level of the target space hierarchy is shown via indentation, classifier features are prefixed with
        (C) and regression features with (R). Additionally, regression features are followed by "= {value}". Every
        feature occurs in its own line, regression features on the same level are sorted alphabatically and always
        preceed the selected classifier on the same level, if any exists.

        :param embedding: the embedding to display
        :param indent: string representing a level of indentation, by default 4 spaces
        :return: a string representing the given embedding
        """
        result = []
        level = self.configuration.hierarchy
        depth = 0
        while level is not None:
            uses = level["_uses"] if "_uses" in level else {}
            for e in sorted(uses):
                f_index = self.configuration.features.index(e)
                if embedding[f_index] >= 0:
                    result.append(f"{indent * depth}(R) {e} = {embedding[f_index]}")
            for f_name in level:
                if f_name.startswith("_"):
                    continue
                f_index = self.configuration.features.index(f_name)
                if embedding[f_index] == 1:
                    level = level[f_name]
                    result.append(f"{indent * depth}(C) {f_name}")
                    break
            else:
                level = None
            depth += 1
        return "\n".join(result)

    @staticmethod
    def _used_features_from(tree):
        result = set()
        if tree is None:
            return result
        for i in range(tree.tree_.n_features):
            if tree.tree_.children_left[i] != tree.tree_.children_right[i]:
                result.add(tree.tree_.feature[i])
        return result

    def get_used_orig_features(self, features=None):
        """
        This method can be used to gather all context words used from the specified features in split nodes.

        :param features: an iterable containing the target space feature names, trees of which should be taken into
                         account; if set to None (default) equal to providing all available feature names
        :return: a numpy array containing the indices of all context words used in a tree's split for any of the
                 specified features
        """
        result = set()
        for t_index in range(len(self.trees)):
            if features is None or self.configuration.features[t_index] in features:
                result.update(self._used_features_from(self.trees[t_index]))

        return np.fromiter(result, np.uint)
