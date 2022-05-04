from math import inf
from random import shuffle

from mysklearn.myutils import *

import mysklearn.myevaluation as myevaluation

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train: list):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        #y_discrete = list(map(self.discretizer, y_train))
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return list(map(self.discretizer, self.regressor.predict(X_test)))

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3, categorical:bool=False):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.categorical = categorical

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        for test_instance in X_test:
            row_distances = []
            row_indexes_dists = []
            for i, train_instance in enumerate(self.X_train):
                dist = 0
                if self.categorical:
                    dist = compute_categorical_distance(train_instance, test_instance)
                else:
                    dist = compute_euclidean_distance(train_instance, test_instance)
                row_distances.append(dist)
                row_indexes_dists.append([i, dist])
            distances.append(dist)
            neighbor_indices.append(row_indexes_dists)
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        _, neighbor_indices = self.kneighbors(X_test)
        y_predicted = []
        for prediction in neighbor_indices:
            min_neighbor = prediction[0]
            for neighbor in prediction:
                if neighbor[1] < min_neighbor[1]:
                    min_neighbor = neighbor
            index = min_neighbor[0]
            y_predicted.append(self.y_train[index])
        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        counts = {}
        for label in y_train:
            counts[label] = counts.get(label, 0) + 1
        self.most_common_label = max(counts, key=counts.get)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.most_common_label for i in X_test]

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        y_labels = list(set(y_train))
        self.priors = {}
        self.posteriors = {}
        for y_label in y_labels:
            self.priors[y_label] = len([i for i in y_train if i == y_label])/len(y_train)
            self.posteriors[y_label] = {}
            X_instances = [X_train[i] for i in range(len(y_train)) if y_train[i] == y_label]
            for attribute in range(len(X_instances[0])):
                att = "att" + str(attribute + 1)
                self.posteriors[y_label][att] = {}
                for X in X_instances:
                    self.posteriors[y_label][att][X[attribute]] = self.posteriors[y_label][att].get(X[attribute], 0) + 1
                for key in self.posteriors[y_label][att]:
                    self.posteriors[y_label][att][key] /= len(X_instances)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for X in X_test:
            curr_label = ""
            max_probability = 0
            for prior in self.priors:
                probability_Ci = self.priors[prior]
                probability_X_given_Ci = 1
                # compute the probability of X given Ci
                for attribute in range(len(X)):
                    att = "att" + str(attribute + 1)
                    probability_X_given_Ci *= self.posteriors[prior][att].get(X[attribute], 0)
                probability_Ci_given_X = probability_X_given_Ci*probability_Ci
                if probability_Ci_given_X > max_probability:
                    max_probability = probability_Ci_given_X
                    curr_label = prior
            y_predicted.append(curr_label)
        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, random_selection: int=None):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.random_selection = random_selection

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # programmatically create a header (e.g. ["att0", "att1",
        # ...] and create an attribute domains dictionary)
        self.header = ["att" + str(i) for i in range(len(X_train[0]))]
        self.attribute_domains = []
        for attribute in range(len(self.header)):
            self.attribute_domains.append(sorted(list(set([X[attribute] for X in X_train]))))
        # next, I advise stitching X_train and y_train together
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # now, making a copy of the header because tdidt()
        # is going to modify the list
        available_attributes = self.header.copy()
        # recall: python is pass by object reference
        self.tree = self.tdidt(train, available_attributes)
        # note: the unit test will assert tree == interview_tree_solution
        # (mind the attribute value order)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for X in X_test:
            curr_node = self.tree
            success = True
            while curr_node[0] != "Leaf":
                attribute = curr_node[1]
                attribute_index = self.header.index(attribute)
                attribute_val = X[attribute_index]
                found_node = False
                for node in curr_node[2:]:
                    if node[1] == attribute_val:
                        curr_node = node[2]
                        found_node = True
                        break
                if found_node is False:
                    # we got to the end of the tree without finding a match
                    # unable to classify instance
                    success = False
                    break
            # now, curr_node is a leaf node
            if success is True:
                y_predicted.append(curr_node[1])
            else:
                y_predicted.append(None)
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            attribute_names = self.header
        rules = self.get_rules(self.tree.copy())
        for rule in rules:
            attributes = rule[0]
            attribute, attribute_value = list(attributes.items())[0]
            attribute_name = attribute_names[self.header.index(attribute)]
            attribute_string = attribute_name + " == " + str(attribute_value)
            for attribute, attribute_value in dict(list(attributes.items())[1:]).items():
                attribute_name = attribute_names[self.header.index(attribute)]
                attribute_string += " AND " + attribute_name + " == " + str(attribute_value)
            label = rule[1]
            print("IF " + attribute_string + " THEN " + class_name + " = " + str(label))

    def get_rules(self, tree):
        """ Get rules from the tree.

        Returns:
            rules(list of list of list of obj): The list of rules.
                Each rule is a list of two items:
                    [0] = dictionary of attributes and values
                    [1] = label
        """
        rules = []
        attribute = tree[1]
        for value in tree[2:]:
            value_rule = {attribute: value[1]}
            if value[2][0] == "Leaf":
                rules.append([value_rule, value[2][1]])
            elif value[2][0] == "Attribute":
                attribute_rules = self.get_rules(value[2])
                updated_attribute_rules = []
                for attribute_rule in attribute_rules:
                    attribute_rule_dict: dict = attribute_rule[0]
                    attribute_rule_class_label = attribute_rule[1]
                    updated_attribute_rule_dict = {**value_rule, **attribute_rule_dict}
                    attribute_rule_dict.update(value_rule)
                    updated_attribute_rules.append([updated_attribute_rule_dict, attribute_rule_class_label])
                rules.extend(updated_attribute_rules)
        return rules

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

    def tdidt(self, current_instances, available_attributes):
        """TDIDT (top down induction of decision tree) algorithm.

        Args:
            current_instances(list of list of obj): The list of training instances (samples).
                The shape of train is (n_train_samples, n_features)
            available_attributes(list of str): The list of available attributes.

        Returns:
            tree(nested list): The extracted tree model.
        """
            # basic approach (uses recursion!!):

        # select an attribute to split on
        attribute = None
        if self.random_selection is not None:
            attribute_set = self.select_random_attributes(available_attributes)
        else:
            attribute_set = available_attributes
        attribute = self.select_attribute(current_instances, attribute_set)
        available_attributes.remove(attribute) # can't split on this again in
        # this subtree
        tree = ["Attribute", attribute] # start to build the tree!!

        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, attribute)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value, att_partition in partitions.items():
            value_subtree = ["Value", att_value]
            #    CASE 1: all class labels of the partition are the same => make a leaf node
            if (len(set([instance[-1] for instance in att_partition])) == 1):
                # TODO: make a leaf node
                leaf_node = ["Leaf", att_partition[0][-1], len(att_partition), len(current_instances)]
                value_subtree.append(leaf_node)
                tree.append(value_subtree)
            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                # TODO: we have a mix of class labels, handle clash w/
                # majority vote leaf node
                att_partition_labels = [instance[-1] for instance in att_partition]
                majority_vote_leaf_value = max(sorted(list([instance[-1] for instance in att_partition])), key=att_partition_labels.count)
                value_subtree.append(["Leaf", majority_vote_leaf_value, len(att_partition), len(current_instances)])
                tree.append(value_subtree)
            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                # TODO: "backtrack" and replace this attribute node
                # with a majority vote leaf node
                values = list(set([instance[-1] for instance in current_instances]))
                leaf_value_counts = {val: len([instance for instance in current_instances if instance[-1]==val]) for val in values}
                majority_vote_leaf_count = max(leaf_value_counts.values())
                majority_vote_leaf_value = max(sorted(leaf_value_counts), key=leaf_value_counts.get)
                tree = ["Leaf", majority_vote_leaf_value, majority_vote_leaf_count, len(current_instances)]
            else: # none of the previous conditions were true... recurse!
                subtree = self.tdidt(att_partition, available_attributes.copy())
                # note the copy
                # append subtree to value_subtree and tree appropriately
                value_subtree.append(subtree)
                tree.append(value_subtree)
        return tree

    def select_attribute(self, instances, attributes):
        """Selects an attribute to split on.

        Args:
            instances(list of list of obj): The list of training instances (samples).
                The shape of train is (n_train_samples, n_features)
            attributes(list of str): The list of available attributes.

        Returns:
            attribute(str): The selected attribute.
        """
        # use entropy to calculate and choose the
        # attribute with the smallest Enew
        # for now, we use random attribute selection
        min_attribute = ""
        min_attribute_entropy = inf
        for attribute in attributes:
            partitions = self.partition_instances(instances, attribute)
            Enew = 0
            for partition in partitions.values():
                Enew += len(partition) / len(instances) * myutils.entropy(partition)
            if Enew < min_attribute_entropy:
                min_attribute = attribute
                min_attribute_entropy = Enew
        return min_attribute

    def select_random_attributes(self, available_attributes):
        """Selects f random attributes from the available attributes.

        Args:
            available_attributes(list of str): The list of available attributes.
            f(int): The number of attributes to select.

        Returns:
            selected_attributes(list of str): The selected attributes.
        """
        selected_attributes = []
        shuffled_attributes = available_attributes.copy()
        shuffle(shuffled_attributes)
        num_attributes = min(self.random_selection, len(available_attributes))
        for i in range(num_attributes):
            selected_attributes.append(shuffled_attributes[i])
        return selected_attributes

    def partition_instances(self, instances, split_attribute):
        """Partitions instances by attribute value.

        Args:
            instances(list of list of obj): The list of training instances (samples).
                The shape of train is (n_train_samples, n_features)
            split_attribute(str): The attribute to partition on.

        Returns:
            partitions(dict): A dictionary of partitions.
                The keys are the attribute values, and the values are the
                list of instances with that attribute value.
        """
        # this is a group by attribute domain
        # let's use a dictionary
        partitions = {} # key (attribute value): value (subtable)
        att_index = self.header.index(split_attribute) # e.g. level -> 0
        att_domain = self.attribute_domains[att_index] # e.g. ["Junior", "Mid", "Senior"]
        for att_value in att_domain:
            partitions[att_value] = []
            # task: finish
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)
        return partitions

    def predict_instance(self, instance):
        """Predicts the class label of a single instance.

        Args:
            instance(list of obj): The instance to predict.

        Returns:
            label(str): The predicted class label.
        """
        curr_node = self.tree
        success = True
        while curr_node[0] != "Leaf":
            attribute = curr_node[1]
            attribute_index = self.header.index(attribute)
            attribute_val = instance[attribute_index]
            found_node = False
            for node in curr_node[2:]:
                if node[1] == attribute_val:
                    curr_node = node[2]
                    found_node = True
                    break
            if found_node is False:
                # we got to the end of the tree without finding a match
                # unable to classify instance
                success = False
                break
        # now, curr_node is a leaf node
        if success is True:
            return curr_node[1]
        else:
            return None



class MyRandomForestClassifier:

    def __init__(self, n_trees, m, f):
        """Initializes a RandomForestClassifier.

        Args:
            n_trees(int): The number of trees in the forest.
            m(int): The number of trees to select from the forest.
        """

        self.n_trees = n_trees
        self.m = m
        self.f = f

    def fit(self, X, y):
        """Trains a random forest classifier.

        1. Generate a random stratified test set consisting of one third of the original data set,
            with the remaining two thirds of the instances forming the "remainder set".
        2. Generate N "random" decision trees using bootstrapping (giving a training and validation set)
            over the remainder set. At each node, build your decision trees by randomly selecting F of
            the remaining attributes as candidates to partition on. This is the standard random forest
            approach discussed in class. Note that to build your decision trees you should still use entropy;
            however, you are selecting from only a (randomly chosen) subset of the available attributes.
        3. Select the M most accurate of the N decision trees using the corresponding validation sets.
        4. Use simple majority voting to predict classes using the M decision trees over the test set.
        """
        trees = {}
        for i in range(self.n_trees):
            X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y)
            classifier = MyDecisionTreeClassifier(random_selection=self.f)
            classifier.fit(X_train, y_train)
            # add accuracy
            trees[classifier] = myevaluation.accuracy_score(classifier.predict(X_test), y_test)
        self.forest = []
        # append the top m trees to the forest
        for i in range(self.m):
            max_accuracy_tree = max(trees, key=trees.get)
            self.forest.append(max_accuracy_tree)
            trees.pop(max_accuracy_tree)

    def predict(self, test):
        """Predicts the class labels for the test instances.

        Args:
            test(list of list of obj): The list of test instances.
                The shape of test is (n_test_samples, n_features)
            forest(list of DecisionTree): The list of decision trees.

        Returns:
            predictions(list of str): The list of class labels.
        """
        predictions = []
        for instance in test:
            instance_predictions = []
            tree: MyDecisionTreeClassifier
            for tree in self.forest:
                instance_predictions.append(tree.predict_instance(instance))
            predictions.append(max(set(instance_predictions), key=instance_predictions.count))
        return predictions