#!/usr/bin/python3

# Author: Sebastian Schmidt
# Date: 14.02.2021

import os
import pickle
import shelve
from typing import Set, List, Callable

import nltk
import numpy as np
from nltk.tokenize import word_tokenize


class COM:
    """
    The COM class is essentially a wrapper for the BigramCollocationFinder contained in NLTK,
    providing some extended functionality and being specifically adapted for use with the Com2Sem model. To get the
    list of features (context words) the Com2Sem model is to be provided, use the method `get_orig_feature_names()`.
    IMPORTANT: For being able to use the COM module, NLTK has to be installed.

    COM objects allow to build several co-occurrence matrices and save them to disk independently, so that the memory
    usage fits the available RAM while still larger corpora can be taken into account. See the constructor documentation
    for more information on this.

    The class supports to construct PMI or PPMI vectors automatically even from individual models and cache them
    using the Python builtin `shelve` package.
    """
    models: List = None
    _limit = -1

    _corpus_path: str = None
    _model_path: str = None

    vocabulary: Set[str] = None
    vocabulary_size: int = 0
    _fixed_vocab = False

    features: Set[str] = None
    number_features: int = 0
    _fixed_features = False

    _feature_mapping = None
    _window_mapping = None
    _tokenizer: Callable = None

    def __init__(self, corpus_path, limit=-1, model_path=None, vocabulary=None, features=None, window_mapping=None,
                 tokenizer=word_tokenize):
        """
        Initializes a new COM object

        :param corpus_path: the file path to the corpus to read (containing plain text in utf-8 compatible encoding)
        :param limit: the maximum amount of characters each built model should take from the corpus; if -1 (default) the
                      whole corpus is read and only one model is constructed (assuming enough memory is available).
        :param model_path: path including a base file name which is used to save all constructed models (suffixed by
                           "_{index}.com"); the models are exported using Python's `pickle` module and thus may take up
                           some space.
        :param vocabulary: can be used to restrict the vocabulary (i.e. the words mapped to the COM's rows); if
                           None (default), all words found in the corpus are used (if `features` is set explicitly, only
                           center words).
        :param features: can be used to restrict the number of features (i.e. the words mapped to the COM's columns); if
                         None (default), all words found in the corpus are added to the feature set (if `vocabulary` is
                         set explictly, only context words).
                         Note: in the case that `vocabulary` as well as `features` are None, they are saved as one and
                         the same object to save memory.
        :param window_mapping: either a positive integer denoting the window radius or a function or lambda
                               expression, which is given all tokens of the preprocessed and tokenized corpus and has to
                               generate an iterable containing seperate windows from them. A window can be defined as
                               one of two formats: a dictionary {"center": "{word}", "window": [...]} or a list where
                               the first element denotes the center word and all others the context. As an example,
                               `lambda tokens: nltk.ngrams(tokens, 2)` is a valid window mapping using the second
                               format. If `window_mapping` is an integer (or None) a centered window is used; if None,
                               only the left and right adjacent context words are taken into account (window radius 1).
        :param tokenizer: a function tokenizing the corpus content (potentially after it has been preprocessed). By
                          default, this is the function `nltk.tokenize.word_tokenize`. The result is passed onto the
                          window mapping.

        NOTE: The model is not built automatically after instanciating; use the `build(...)` method explicitly for this
              purpose.
        """
        self.models = []
        self._corpus_path = corpus_path
        self._limit = limit
        self._model_path = model_path

        self.vocabulary = set(vocabulary) if vocabulary else set()
        if vocabulary is not None:
            self._fixed_vocab = True

        self.features = set(features) if features else (set() if vocabulary is not None else self.vocabulary)
        if features is not None:
            self._fixed_features = True

        self._feature_mapping = {}
        self._window_mapping = window_mapping or 1
        self._tokenizer = tokenizer

        self._bigram_measure = nltk.collocations.BigramAssocMeasures()
        self.pmi = self._bigram_measure.pmi

    def build(self, start_index=0, end_index=-1, skip_index=None, **kwargs):
        """
        Use this method to build the co-occurrence matrices. The results are stored either as an property of the
        object, or as an pickle-export on the hard drive if a `model_path` is defined for the object. By the `limit`
        parameter specified for the instanciation it is selected how many models are built for the corpus: if you
        have a corpus with 1 GB of plain text and set limit to 500_000_000 characters (~ 500 MB), you will end up with
        two models each trained on the first and the second half of the corpus.

        IMPORTANT: Rebuilding a corpus without customizing the `start_index` parameter will lead to overwriting the
        existing models.

        :param start_index: if one wants to replace only specific trained models, `start_index` can be handy. Only
                            models with an index greater or equal than `start_index` are replaced in the process of
                            building (if no already trained models are available anymore, new ones are added).
        :param end_index: the method stops with building models as this index is reached (does not stop until the corpus
                          has been processed completely for `end_index=-1`). Can be helpful if one only wants to build
                          a co-occurrence matrix on a port of the data, e.g.
                          `COM("corpus.txt", 1_000_000).build(end_index=0)` will build the matrix on only 1 MB of the
                          full corpus.
        :param skip_index: an iterable containing indices which should not be replaced; if None (default) treated as
                           empty
        :param kwargs: may contain two optional parameters: `preprocessor` should be a function or lambda expressing
                       transforming the corpus read, so that it can be passed on to the tokenizer (e.g. convert to
                       lowercase, remove special characters etc.). Usually, it should output a string as well. `filter`
                       is applied to each built BigramCollocationFinder after it has been built, e.g. to remove all
                       matrix entries not adding enough information. Given a COM-object com, the filter parameter might
                       look as follows: `filter=(com.pmi, 0)` where the first component defines the metric applied to
                       the BigramCollocationFinder (refer to its `apply_ngram_filter` function) and the threshold that
                       has to be met for a pair of words to be kept in the matrix. It is advised to use the filter if
                       only one model should be trained, with a threshold of at least 0, to save memory. That of course
                       means that one retains a PPMI matrix, as a matter of fact.
        :return: the object itself
        """
        with open(self._corpus_path, encoding="utf-8") as file:
            te = file.read(self._limit)
            index = start_index if start_index >= 0 else len(self.models)
            while te != "":
                if 0 <= end_index < index:
                    break
                if skip_index is None or index not in skip_index:
                    self._build(te, index, **kwargs)
                te = file.read(self._limit)
                index += 1
        self.vocabulary_size = len(self.vocabulary)
        self.number_features = len(self.features)
        for word, i in zip(sorted(self.features), range(self.number_features)):
            self._feature_mapping[word] = i

        return self

    def get_orig_feature_names(self):
        """
        Returns an alphabetically sorted list of the used features (i.e. context words).

        :return: the list of used features applicable e.g. for the Com2Sem constructor
        """
        return sorted(self.features)

    def _get_contexts(self, tokens):
        if type(self._window_mapping) == int and self._window_mapping > 0:
            return nltk.ngrams(tokens, 2 * self._window_mapping + 1, pad_left=True, pad_right=True)
        elif callable(self._window_mapping):
            return self._window_mapping(tokens)
        else:
            raise ValueError("window_mapping must be either a positive integer denoting the window size or a function!")

    def _build(self, corpus, index, **kwargs):
        if "preprocessor" in kwargs:
            corpus = kwargs["preprocessor"](corpus)
        tokens = self._tokenizer(corpus)
        wd = nltk.FreqDist()
        cwd = nltk.FreqDist()
        feature_set = set()
        for window in self._get_contexts(tokens):
            if type(window) == dict:
                center = window["center"]
                window = window["window"]
            else:
                center_index = self._window_mapping if type(self._window_mapping) == int else 0
                center = window[center_index]
                window = window[:center_index] + window[center_index + 1:]
            if center is None or (self._fixed_vocab and center not in self.vocabulary):
                continue
            wd[center] += 1
            for i in range(0, len(window)):
                if window[i] is None or (self._fixed_features and window[i] not in self.features):
                    continue
                cwd[(center, window[i])] += 1
                feature_set.add(window[i])
        bcf = nltk.collocations.BigramCollocationFinder(wd, cwd)

        if "filter" in kwargs:

            def filter_function(w1, w2):
                try:
                    return (bcf.score_ngram(kwargs["filter"][0], w1, w2) or 0) <= kwargs["filter"][1]
                except ZeroDivisionError:
                    # For some reason a DivisionByZero occurs here sometimes ... So let's use the Holzhammer method :-)
                    return False

            bcf.apply_ngram_filter(
                filter_function
            )

        if self._model_path is not None:
            with open(self._model_path + f"_{index}.com", "wb") as file:
                pickle.dump(bcf, file)
            if index >= len(self.models):
                self.models.append(self._model_path + f"_{index}.com")
        else:
            if index >= len(self.models):
                self.models.append(bcf)
            else:
                self.models[index] = bcf

        if not self._fixed_features:
            self.features.update(feature_set)
        if not self._fixed_vocab:
            self.vocabulary.update(wd.keys())

    def _load(self, index):
        if self._model_path is not None:
            file_path = (self._model_path + f"_{index}.com") if type(index) == int else index
            with open(file_path, "rb") as file:
                bcf = pickle.load(file)
            return bcf
        else:
            return self.models[index]

    def _gather_stats(self, index, words, vec, p_i, p_j, exclude_indices):
        bcf = self._load(index)
        nix = 0
        for word in words:
            if nix in exclude_indices:
                continue
            overall_word = 0
            for context_word, i in self._feature_mapping.items():
                co_count, [overall_word, overall_ctxt] = bcf.score_ngram(lambda *stats: stats[:2], word,
                                                                         context_word) or (0, [0, 0])
                vec[nix][i] += int(co_count)
                p_j[nix][i] += overall_ctxt
            p_i[nix] += overall_word
            nix += 1
        return bcf.N

    def _get_bcf_pmi(self, words, index, ppmi, shelve_db: shelve.Shelf):
        vec = np.zeros((len(words), self.number_features), np.int)
        nix = 0
        for word in words:
            if shelve_db is not None and word in shelve_db:
                vec[nix] = shelve_db[word]
                nix += 1
                continue
            for context_word, i in self._feature_mapping.items():
                vec[nix][i] = np.max((
                    (self.models[index].score_ngram(self._bigram_measure.pmi, word, context_word) or 0) * 1000,
                    0 if ppmi else -np.infty
                ))
            nix += 1
        return vec.astype(np.ushort) if ppmi else vec

    def get_vectors_raw(self, words: List[str], shelve_db: shelve.Shelf = None, full=False, **kwargs) -> np.ndarray:
        """
        Use this method to calculate the raw-count-based co-occurrence vectors for all given words.

        :param words: a list of words for which the co-occurrence vectors should be calculated
        :param shelve_db: if a shelf is supplied, all vectors already contained in it are re-used instead of
                          calculating them again. Additionally, all vectors that are not yet cached are added to
                          the shelf right after construction.
        :param full: if True (NOT by default) returns a tuple with additional counts that can be used to calculate other
                     more advanced metrics
        :return: if `full` is False (default): numpy.ndarray containing all co-occurrence vectors as a row according to
                 the original list's order. If true, a four-tuple containing the co-occurrence vectors, the row-sums of
                 the co-occurrence matrix, the column-sums (per word) of the co-occurrence matrix and the overall number
                 of detected co-occurrances. The column-sums are collected per word (i.e. in a matrix) because the
                 `BigramCollocationFinder` returns 0 if two words never appear together, and as such it is not always
                 the actual sum.
        """
        vec = np.zeros((len(words), self.number_features,), np.uint)

        p_i = np.zeros((len(words),))
        p_j = np.zeros((len(words), self.number_features,), np.uint)

        abs_d = 0

        exclude_indices = [] if "exclude_indices" not in kwargs else kwargs["exclude_indices"]

        if shelve_db is not None:
            for i in range(len(words)):
                if words[i] in shelve_db:
                    vec[i] = shelve_db[words[i]]
                    exclude_indices.append(i)

        if self._model_path:
            if len(self.models) == 0:
                index = 0
                while os.path.exists(self._model_path + f"_{index}.com"):
                    abs_d += self._gather_stats(index, words, vec, p_i, p_j, exclude_indices)
                    index += 1
            else:
                for file in self.models:
                    if os.path.exists(file):
                        abs_d += self._gather_stats(file, words, vec, p_i, p_j, exclude_indices)
        else:
            for index in range(len(self.models)):
                abs_d += self._gather_stats(index, words, vec, p_i, p_j, exclude_indices)

        if shelve_db is not None:
            for nix in range(len(words)):
                if nix in exclude_indices:
                    continue
                shelve_db[words[nix]] = vec[nix]

        return vec if not full else (vec, p_i, p_j, abs_d)

    def get_vectors_pmi(self, words: List[str], ppmi=True, shelve_db: shelve.Shelf = None) -> np.ndarray:
        """
        Use this method to calculate the pmi-based co-occurrence vectors for all given words.

        :param words: a list of words for which the (P)PMI vectors should be calculated
        :param ppmi: if True (default) the returned vectors are in PPMI format, thus negative values are replaced with 0
        :param shelve_db: if a shelf is supplied, all vectors already contained in it are re-used instead of
                          calculating them again. Additionally, all vectors that are not yet cached are added to
                          the shelf right after construction.
        :return: a numpy.array containing all co-occurrence vectors as a row according to the original list's order
        """
        if len(self.models) == 1:
            # The builtin-PMI-calculation is more efficient, so if possible we want to use that
            return self._get_bcf_pmi(words, index=0, ppmi=ppmi, shelve_db=shelve_db)

        exclude_indices = []

        if shelve_db is not None:
            for i in range(len(words)):
                if words[i] in shelve_db:
                    exclude_indices.append(i)

        vec, p_i, p_j, abs_d = self.get_vectors_raw(words, full=True, exclude_indices=exclude_indices)

        for nix in range(len(words)):
            if p_i[nix] == 0:
                continue
            elif nix in exclude_indices:
                vec[nix] = shelve_db[words[nix]]
                continue

            for j in range(self.number_features):
                if vec[nix][j] == 0:
                    continue
                vec[nix][j] = np.max((
                    self._bigram_measure.pmi(vec[nix][j], (p_i[nix], p_j[nix][j]), abs_d) * 1000,
                    0 if ppmi else -np.infty
                ))
            if shelve_db is not None:
                shelve_db[words[nix]] = vec[nix]
        return vec.astype(np.ushort) if ppmi else vec

    def get_single_vector_ppmi(self, word: str, shelve_db: shelve.Shelf = None):
        """
        Returns the PPMI-based co-occurrence vector for the given word or False if it contains no information.

        :param word: the word for which the co-occurrence vector should be calculated
        :param shelve_db: a shelf which is used to cache the respective word (as such avoids calculating a
                          vector several times) or None (default)
        :return: a vector if the word is contained in the corpus and carries information (i.e. non-zero PMI values) or
                 False otherwise. This method can be used as `co_occurrence_lookup` function to the Com2Sem class.
        """
        res = self.get_vectors_pmi([word], shelve_db=shelve_db)[0]
        if res.max() == 0:
            return False
        return res
