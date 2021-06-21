# Com2Sem
This project is an implementation created along with [my Bachelor's thesis](https://www-ai.cs.tu-dortmund.de/PublicPublicationFiles/schmidt_2021a.pdf) about a new word embedding approach based on
decision trees. Its main objectives are explainability and interpretability, which, however, lead to some technical and
conceptional limitations (at least for the time being); so if you are looking for a powerful and easily set up embedding
model, this is probably not an ideal choice. If you are still interested: in the following, it will shortly be outlined
what this project is all about.

## The Distributional Hypothesis in a Nutshell
For an in-depth explanation of the Distributional Hypothesis/Distributional Semantics
refer to [Wikipedia](https://en.wikipedia.org/wiki/Distributional_semantics), however, the principles are important
to understand for the approach, so here is a short overview.

Given the example text: "The lion has escaped from the zoo. The lion then followed the zebra to Grand Central Station."

The example sentence makes use of a specific vocabulary, in this case:

V = {"and", "escaped", "followed", "from", "Grand Central Station", "has", "lion", "the", "then", "to", "zebra", "zoo"}

If we define under which circumstances two words in a text are connected, we can furthermore determine a word
distribution. For our example, say that word A is connected to word B if word B directly preceeds it:

- \#(a, b) = 2 for `"lion" -> "the"`

- \#(a, b) = 1 for among others `"has" -> "lion"`, `"escaped" -> "has"`, `"from" -> "escaped"`, `"zoo" -> "the"`, `"zebra" -> "the"`

Now, the distributional hypothesis states that two words are semantically related if they possess similar distributions.
This is already visible in our small example: note that all words having a positive distribution with "the" are nouns
(unsurprisingly). Generally thinking about language use, one could easily come up with more examples:

- "to eat" will usually only appear with living beings and food
- words occurring frequently with "from", "in", "to", "at" or similar prepositions will mostly describe locations
- "who" follows words describing a human in most cases
- ...

To make use of these properties, it is common to represent words in a distributional space containing all the information of
how often a word appears with others in all its contexts, and we assume the vectors we get out of this space
can be used to solve different NLP (Natural Language Processing) tasks.

## More on Embeddings
For a distributional space, one often constructs a so-called co-occurrence matrix (COM) where rows and columns represent
words and every cell contains a value that measures the strength of their relationship. The measure may be the raw count
of co-occurrences, or more meaningful metrics like [PMI](https://en.wikipedia.org/wiki/Pointwise_mutual_information).

COMs, however, get large very quickly, and perhaps we do not actually need *all* of the values. Several solutions have
already been worked out to compress the distributional space without loosing relevant information; refer e.g. to
[Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) (LSA) or neural embeddings like
[word2vec](https://en.wikipedia.org/wiki/Word2vec).

Here is the problem: by transforming the distributional space, we generally loose knowledge of what information our
embeddings actually contain. If we beforehand could have said: "Sure, `from`, `in`, `to` and `at` all have similarly high
co-occurrence values, that's why `Paris` and `Berlin` are more closely related then `Paris` and `stone`", this information is
now contained in any (or possibly distributed over several) of the space's dimensions. This could be dangerous, if an
embedding accidentally learns stereotypical associations, and we have no chance of noticing that.

## Finding Discriminative Context Words
So, what could we do about that? Let us summarize what we want to achieve:

- we want to know which word contexts exactly contribute to a decision that is based on our model (e.g. similarity of
  different words)
  
- the resulting embeddings should not be too large, so we save memory

- optimally, the resulting embeddings are already interpretable by humans, e.g. by looking at the embedding for "baker"
  we would directly see: the word defines a human and a profession, and perhaps even "is related to bread" or similar

As an approach for the last point, we could define a new embedding space not being distributionally, but semantically
motivated -- not a new invention either. For that, we would need a mapping of the old to the new space; a mapping that also
takes our first point into account.

To learn a mapping to a space we defined ourselves, obviously training data is needed. Thus, in effect, we need a
mapping approach that lets us see on which contexts it bases its classification, and to do that, it must find
discriminative context words that separate training data we labeled as a specific class, and all other training data.

Which leads us to: decision trees. And, so far already, we covered the motivation, background and even some useful
terms for this package!


## Using Com2Sem
### What does it do?
Com2Sem implements an approach for Word Embedding where co-occurrence vectors are mapped to a space of semantical
categories defined by humans. The mapping is achieved explainably, interpretably and transparently via decision trees
learned directly on the original co-occurrence vectors.

So far, the whole approach and especially the implementation are rather experimental and could in theory be used in many
different ways, the best of which are not yet discovered. A semantic target space manually designed for testing purposes
and still not in any final version is included in this repository as well, along with some training data.

The following sections will provide the basic knowledge and a few examples on how this module's content could be used.

### Dependencies & Setup
This project is a [Python](https://www.python.org/) module, built for and with Python 3.7; it could work with earlier
versions from Python 3.5 on, but that has not been tested.

Some non-standard libraries are necessary to use the functionality contained in the module or parts of it:

- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [PyYAML](https://pypi.org/project/PyYAML/)
- For being able to use the `Com2Sem.justify` method or the `COM` class: [NLTK](https://www.nltk.org/)

The module can be used by placing its folder inside your project, or even more comfortably inside your Python
installation's site-packages folder, making it available from everywhere. Therefore, a `setup.cfg` and `setup.py` script
have been added which use Python's `setuptools` package to automate the installation. This is, however, not necessary.

### Building a Co-Occurrence Matrix
For constructing co-occurrence matrices, a wrapper class for the NLTK class `BigramCollocationFinder` is provided,
namely `COM`. The documentation is already quite excessive, so here just a few examples.

#### Minimal Working Example
```python
from com2sem.com import COM
CORPUS_FILE = '/path/to/corpus.txt'

com = COM(CORPUS_FILE).build()

single_ppmi_vector = com.get_single_vector_ppmi("computer")
ppmi_vector_matrix = com.get_vectors_pmi(["computer", "science"])

assert (ppmi_vector_matrix[0] == single_ppmi_vector).all()
```
#### Specify a Custom Window Radius
```python
[...]
WINDOW_RADIUS = 5

com = COM(CORPUS_FILE, window_mapping=WINDOW_RADIUS).build()
```
#### Example with High Customization
```python
[...]

com = COM(
    CORPUS_FILE, limit=100_000_000, model_path="./coms/com_base_name",
    vocabulary=LIST_OF_NOUNS, features=LIST_OF_ADJECTIVES,
    window_mapping=lambda tokens: map(reversed, nltk.ngrams(token, 2)),
    tokenizer=lambda text: re.split("\\s+", text)
)
com.build(start_index=1, end_index=3, preprocessor=lambda text: text.lower(), filter=(com.pmi, 0))
```
To summarize what happens here:
- The corpus is configured to read in chunks of 100 million characters (defined by `limit`) that are processed independently
  (i.e. technically generate their own co-occurrence matrix).
- As rows of the co-occurrence matrices, only the words contained in `LIST_OF_NOUNS` are allowed and as
  columns only those in `LIST_OF_ADJECTIVES`. Also, we increase the co-occurrence count for word A w.r.t. B only if they
  match the pattern "B A"; i.e. a window always consists of two words where the right one is seen as the center word.
- The text is tokenized by simply splitting it at every coherent sequence of whitespace
- The co-occurrence matrix is not collected on the full corpus, but only from the index 100 million to 400 million (the
  second, third and fourth chunk of the corpus).
- The text is converted to lower-case before tokenization.
- After construction, every combination of words is filtered out for any sub-model if their PMI score is not greater
  than 0 (i.e. their occurrences in the corpus would be correlated).
- Each matrix is not retained in memory but exported to a `.com` file generated from the configured `model_path`
  attribute and the model's respective index (in effect, the `.com` file is just a pickle dump of the
  `BigramCollocationFinder`).

### Configuring a Target Space
For the Com2Sem model to produce an embedding, it needs to know in which space to map the co-occurrence vectors. This
space will be called _Target Space_ in the following. As mentioned in the third section, the target space should be
human-readable, so it is also to be defined manually; for the Com2Sem model, the target space structure is provided by
a configuration YAML where nested dictionaries contain features as keys.

For example, we could train a space finding animals and animal subspecies. For that, we could use the following space:

* is animal
  * is amphibian
  * is arachnid
  * is bird
  * is fish
  * is insect
  * is mammal
  * is reptile
  
By defining a hierarchy of features, we may achieve three advantages:
- trees classifying a specialized concept do not have to repeat conditions that discriminate a more general meaning aspect
  if their parent tree handles this already, which leads to better readability/interpretability
- we avoid embeddings containing two totally unrelated concepts at the same time (`has teeth` and `is a mountain` for example)
- we can trace multiple routes inside the tree if the classifiers show enough evidence; i.e. gain the possibility to
  recognize homonymy
  
Target spaces may also contain _regression features_, features with potentially continuous values between 0 and 1; these
are also embedded in the hierarchy, but cannot contain child features. Furthermore, they are not thoroughly tested so far,
so not much can be said about their performance.

Refer to the documentation in `conf.py` for more information.

**NOTE:** The target space configuration has to be written manually; until now, there is no API allowing to automatically
modify a target space configuration file or to assist in adapting training data.

### Designing Training Data
Training data must be supplied as a CSV file, where the first column contains words/tokens as string and the subsequent
columns represent the target space dimensions. The target space dimensions are all (classifier **and** regression)
features sorted alphabetically by their names. Following values are allowed:

- classifier features: -1 (undefined), 1 (selected)
- regression features: -1 (undefined), 0 to 1 including both end points

_NOTE:_ The meaning of regression features should be user-defined, it could be some fuzzy property (0: not applicable at
all, 1: certainly applicable) or a quantification (0: never, 1: always) .

For creating and managing training data, this repository contains a Tkinter-based tool `TrainingDataCreator`, providing
a graphical user interface for this purpose.

You can start it e.g. via `python tdc.py`. To use the tool, only the PyYAML package is required, all other used packages
are usually built-in (tkinter, csv).

When the TrainingDataCreator is started, a system file dialog should open asking to select a target space configuration
file. After one has been selected, the target space classifier features can be selected in a TreeView widget on the
bottom-left. The frame at the top will show all added words along with some of their defined properties (regression
features); double-clicking on one of the words will lead to loading the respective entry by selecting the respective
most specific classifier feature and setting all entries on the bottom-right frame to the saved values.

On the bottom-right, one will see a text field where a word/token to be added can be specified, and beneath that
available regression features will appear once a category is selected offering them. By using the buttons "Add",
"Modify" or "Remove", the training data can be expanded, modified or reduced.

To save or load your training data, use "Ctrl-s" and "Ctrl-o", respectively; as an alternative, use the "File" menu at
the top of the window.

### Using the Com2Sem model
As the documentation already covers all the details, this section will present a few pieces
of example code.

### Minimum Working Example
```python
from com2sem import Com2Sem

[...] # Build a co-occurrence matrix `com: COM`

CONFIG_PATH = "/path/to/target-space-configuration.yaml"
TRAINING_DATA_PATH = "/path/to/training-data.csv"

model = Com2Sem(com.get_orig_feature_names(), CONFIG_PATH)
model.train(TRAINING_DATA_PATH, com.get_single_vector_ppmi)
```

### Using the Model
```python
[...] # Model `model` and COM `com` already defined

embeddings = model.predict(com.get_single_vector_ppmi("computer"))
assert type(embeddings) == np.ndarray and embeddings.shape[1] == len(model.configuration.features)

# Print a nice representation of all embeddings:
for e in embeddings:
    print(model.render_embedding(e))
    print("----------------")

# Show all context words that influence how "computer" is classified w.r.t. any feature

with open(CORPUS_FILE) as file:
    corpus_text = file.read()

for context, relevant_words in model.justify("computer", com.get_single_vector_ppmi("computer"),
                                             model.configuration.features, corpus_text):
    print("Context:", " ".join(context))
    for discriminating_word, affected_feature in relevant_words:
        print("Predicted as", affected_feature, "because of", discriminating_word)

# Calculate similarity
print(
    "Similarity of computer and PC:",
    model.normalized_concept_overlapping(
        com.get_single_vector_ppmi("computer"),
        com.get_single_vector_ppmi("PC"),
        False
    )
)

# Display a decision diagram of the tree for the `is digital` feature
model.show_tree("is digital")
```

## Target Space for Testing Purposes
This repository also contains a predefined target space configuration file
`semantic_space.yaml` and some training data `semantic_space.csv`; however, I **strongly** discourage using those files
for reasons other than testing the Com2Sem model, as they are not the result of a high-quality study. They were created
by the example of [MultiNet's](https://en.wikipedia.org/wiki/MultiNet) semantical structures solely for testing purposes and a proof-of-concept.

Some general problems of structure and dataset:
- the dataset is not complete, there are features defined having no or only few examples
- some assignments are questionable; I for the most part decided against fixing them, so the results in my thesis were
  recreatable
- this space did not perform too well in tests, leading to the assumption that the structure is not suitable for a
  model of natural language (at least the English language)
  
Another improved target space and/or dataset may be added in the future.
