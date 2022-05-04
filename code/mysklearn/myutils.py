import numpy as np
import mysklearn.myevaluation as myevaluation

def compute_euclidean_distance(v1, v2):
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def compute_categorical_distance(v1, v2):
    for i in range(len(v1)):
        if v1[i] != v2[i]:
            return 1
    return 0

def get_samples_from_folds(X, fold):
    return [X[i] for i in fold]

def normalize(data_list):
    return [float(i)/max(data_list) for i in data_list]

def rank_mpgs(mpg_list):
    rankings = []
    for mpg in mpg_list:
        if mpg <= 13:
            rankings.append(1)
        elif mpg == 14:
            rankings.append(2)
        elif 15 <= mpg <= 16:
            rankings.append(3)
        elif 17 <= mpg <= 19:
            rankings.append(4)
        elif 20 <= mpg <= 23:
            rankings.append(5)
        elif 24 <= mpg <= 26:
            rankings.append(6)
        elif 27 <= mpg <= 30:
            rankings.append(7)
        elif 31 <= mpg <= 36:
            rankings.append(8)
        elif 37 <= mpg <= 44:
            rankings.append(9)
        else:
            rankings.append(10)
    return rankings

def entropy(partition):
    """Compute the entropy of a partition.

    Args:
        partition (list of list): input partition

    Returns:
        entropy: entropy of the partition
    """
    class_labels = list(set([instance[-1] for instance in partition]))
    class_proportions = [len([instance for instance in partition if instance[-1] == label])/len(partition) for label in class_labels]
    return -sum([p * np.log2(p) for p in class_proportions])

def evaluate_model(classifier, X, y):
    """Evaluates the model by computing the accuracy of the classifier
        with k-fold stratified cross validation"""
    X_train_folds, X_test_folds = myevaluation.stratified_kfold_cross_validation(X, y, 10, shuffle=True)
    y_test = []
    y_pred = []
    for fold in range(len(X_train_folds)):
        X_train = [X[i] for i in X_train_folds[fold]]
        y_train = [y[i] for i in X_train_folds[fold]]
        X_test = [X[i] for i in X_test_folds[fold]]
        y_fold_test = [y[i] for i in X_test_folds[fold]]
        classifier.fit(X_train, y_train)
        y_fold_pred = classifier.predict(X_test)
        y_test.extend(y_fold_test)
        y_pred.extend(y_fold_pred)
    accuracy = myevaluation.accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Error rate:", 1 - accuracy)

def bin_data(data_column, bin_size = 10):
    minv = min(data_column)
    maxv = max(data_column)
    binned_col = []
    for d in data_column:
        b = int((d-minv) / (maxv - minv) * bin_size)
        binned_col.append(b)
    return binned_col

def bin_all(X):
    binned_X_cols = []
    for col in range(len(X[0])):
        column = [row[col] for row in X]
        binned = bin_data(column)
        binned_X_cols.append(binned)
    binned_X = []
    for i in range(len(binned_X_cols)):
        row = [item[i] for item in binned_X_cols]
        binned_X.append(row)
    return binned_X