#!/usr/bin/python3

# Author: Sebastian Schmidt
# Date: 14.02.2021

import re
import yaml


class Configuration:
    """
    This class can be used to load a target space configuration file, which must be specified in the
    YAML format (https://yaml.org/). Changing the configuration file programmatically is currently not supported.

    As an overview to the configuration format (for a quick introduction refer to the README):
      - the overall configuration is one nested dictionary where every key denotes a classifier feature
      - the feature hierarchy is defined by their position in the tree structure, however, the feature _order_ of the
        space is determined by their name (sorted alphabetically)
      - the special "_uses" key introduces regression features into the space, allowing continuous values from 0 to 1
        including both endpoints, and additionally -1 to denote "unknown". Subordinate to a `_uses` key a list of
        feature names can be added, their position is defined by the position of the `_uses` key in the hierarchy.
      - regressor values can be fixed using "=", this will lead to the respective features always being predicted as the
        value after the equal sign for all embeddings selecting the parent feature.
      - by using the "=~" operator, the Training Data Creator can be instructed to set the respective value as a default
        value; however, it makes no difference in actual prediction.

    As an example:

    ```
    _uses:
        - movable =~ 1  # -> 1 is set as a default for the movable feature in TDC
    animal:
        _uses:
            - living on land # new regression feature introduced, -1 by default
        mammal: "-"
        bird: "-"
        fish: "-"
    plant:
        _uses:
            - movable = 0
        tree: "-"
    ```
    """

    hierarchy = None
    features = None
    configuration_file = None

    def __init__(self, configuration_file):
        """
        Initializes a new Configuration object.

        :param configuration_file: path to the YAML file from which the configuration should be loaded

        IMPORTANT: The configuration is NOT loaded automatically; use the `build()` method explicitly for that purpose.
        """
        self.configuration_file = configuration_file

    def build(self):
        """
        Loads the configuration, constructing the `hierarchy` property and the `features` list.
        Note that the `features` list, providing the index mapping for the target space, is sorted
        alphabetically from all used classifier and regression features.

        :return: void
        """
        with open(self.configuration_file, encoding="utf-8") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        features = set()
        self.hierarchy = self._build(config, features=features)
        self.features = sorted(features)

    def _build(self, level: dict, features):
        hierarchy_level = {}
        for e, v in level.items():
            if not e.startswith("_"):
                features.add(e)
            else:
                hierarchy_level[e] = {}
                for i in v:
                    i = re.split("\\s*=\\s*", i, maxsplit=1)
                    if len(i) == 2:
                        name = i[0]
                        if i[1].startswith("~"):
                            value = (float(i[1][1:].strip()), False)
                        else:
                            value = (float(i[1]), True)
                    else:
                        name, value = i[0], None
                    hierarchy_level[e][name] = value
                    features.add(name)
                continue
            if type(v) == str:
                hierarchy_level[e] = None
            else:
                hierarchy_level[e] = self._build(v, features)
        return hierarchy_level
