import numpy as np

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