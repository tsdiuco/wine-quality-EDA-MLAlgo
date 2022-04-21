from mysklearn import myutils

from math import ceil
from mysklearn import myutils
import numpy.random as random

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    rng = random.default_rng(random_state)
    if shuffle:
        indices = list(range(len(X)))
        rng.shuffle(indices)
        X = [X[i] for i in indices]
        y = [y[i] for i in indices]
    num_test_instances = test_size
    if type(test_size) == float:
        num_test_instances = ceil(test_size*len(X))
    X_train = X[:len(X)-num_test_instances]
    X_test = X[len(X)-num_test_instances:]
    y_train = y[:len(y)-num_test_instances]
    y_test = y[len(y)-num_test_instances:]
    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    rng = random.default_rng(random_state)
    indices = list(range(len(X)))
    if shuffle:
        rng.shuffle(indices)
    fold_size = ceil(len(X) / n_splits)
    X_train_folds = []
    X_test_folds = []
    index = 0
    for i in range(n_splits):
        X_test_fold = []
        while len(X_test_fold) < fold_size and index < len(X):
            X_test_fold.append(indices[index])
            index += 1
        X_test_folds.append(X_test_fold)
        X_train_fold = [i for i in range(len(X)) if i not in X_test_fold]
        X_train_folds.append(X_train_fold)
    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    rng = random.default_rng(random_state)
    if shuffle:
        indices = list(range(len(X)))
        rng.shuffle(indices)
        X = [X[i] for i in indices]
        y = [y[i] for i in indices]
    # partition based on y labels
    partitions = {}
    for i in range(len(y)):
        if y[i] in partitions:
            partitions[y[i]].append(i)
        else:
            partitions[y[i]] = [i]
    # Create folds
    folds = []
    for i in range(n_splits):
        folds.append([])
    i = 0
    for partition in partitions:
        for index in partitions[partition]:
            folds[i%n_splits].append(index)
            i += 1
    # Separate train and test
    X_train_folds = []
    X_test_folds = []
    for i in range(n_splits):
        X_test_folds.append(folds[i])
        # append the rest of the indices
        X_train_fold = [x for x in range(len(X)) if x not in folds[i]]
        X_train_folds.append(X_train_fold)
    return X_train_folds, X_test_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    random.seed(random_state)
    if n_samples is None:
        n_samples = len(X)
    X_sample = []
    y_sample = []
    indices = list(range(len(X)))
    rand_indices = random.choice(indices, n_samples)
    for rand_index in rand_indices:
        X_sample.append(X[rand_index])
        if y is not None:
            y_sample.append(y[rand_index])
    X_out_of_bag = [sample for sample in X if sample not in X_sample]
    if y is not None:
        y_out_of_bag = [sample for sample in y if sample not in y_sample]
    else:
        y_sample = None
        y_out_of_bag = None
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    confusion_matrix = []
    for i in range(len(labels)):
        confusion_matrix.append([labels[i]])
        for j in range(len(labels)):
            count = 0
            for sample in range(len(y_true)):
                if y_true[sample] == labels[i] and y_pred[sample] == labels[j]:
                    count += 1
            confusion_matrix[i].append(count)
    return confusion_matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    num_correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            num_correct += 1
    if normalize:
        return num_correct/len(y_true)
    return num_correct


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    tp = 0
    fp = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i] and y_pred[i] == pos_label:
            tp += 1
        if y_pred[i] != y_true[i] and y_pred[i] == pos_label:
            fp += 1
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    tp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i] and y_pred[i] == pos_label:
            tp += 1
        if y_pred[i] != y_true[i] and y_pred[i] != pos_label:
            fn += 1
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def classification_report(y_true, y_pred, labels=None, output_dict: bool = False):
    """Build a text report and a dictionary showing the main classification metrics.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        output_dict(bool): If True, return output as dict instead of a str

    Returns:
        report(str or dict): Text summary of the precision, recall, F1 score for each class.
            Dictionary returned if output_dict is True. Dictionary has the following structure:
                {'label 1': {'precision':0.5,
                            'recall':1.0,
                            'f1-score':0.67,
                            'support':1},
                'label 2': { ... },
                ...
                }
            The reported averages include macro average (averaging the unweighted mean per label) and
            weighted average (averaging the support-weighted mean per label).
            Micro average (averaging the total true positives, false negatives and false positives)
            multi-class with a subset of classes, because it corresponds to accuracy otherwise
            and would be the same for all metrics. 

    Notes:
        Loosely based on sklearn's classification_report():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    results_dict = {}
    for label in labels:
        output_dict[label] = {}
        output_dict[label]['precision'] = binary_precision_score(y_true, y_pred, labels, label)
        output_dict[label]['recall'] = binary_recall_score(y_true, y_pred, labels, label)
        output_dict[label]['f1-score'] = binary_f1_score(y_true, y_pred, labels, label)
    if output_dict:
        return results_dict
    else:
        return str(results_dict)

def classification_output(X, y, classifier, labels, pos_label):
    X_train_folds, X_test_folds = stratified_kfold_cross_validation(X, y, n_splits=10)
    accuracy = 0
    y_test = []
    y_pred = []
    for fold in range(len(X_train_folds)):
        X_train = [X[i] for i in X_train_folds[fold]]
        y_train = [y[i] for i in X_train_folds[fold]]
        X_test = [X[i] for i in X_test_folds[fold]]
        y_fold_test = [y[i] for i in X_test_folds[fold]]
        classifier.fit(X_train, y_train)
        y_fold_pred = classifier.predict(X_test)
        accuracy += accuracy_score(y_fold_pred, y_fold_test)
        y_test.extend(y_fold_test)
        y_pred.extend(y_fold_pred)
    accuracy /= len(X_train_folds)
    print("Accuracy:", accuracy)
    print("Error rate:", 1 - accuracy)
    print("Precision:", binary_precision_score(y_test, y_pred, labels=['A', 'H'], pos_label='H'))
    print("Recall:", binary_recall_score(y_test, y_pred, labels=['A', 'H'], pos_label='H'))
    print("F1 score:", binary_f1_score(y_test, y_pred, labels=['A', 'H'], pos_label='H'))