from mysklearn.myclassifiers import MyKNeighborsClassifier
from mysklearn.mypytable import MyPyTable
import mysklearn.myutils as myutils

import pickle
import os

def get_training_sets(data):
    X = data.get_columns(['fixed acidity', 'citric acid', 'residual sugar', 'pH', 'sulphates', 'alcohol'])
    y = data.get_column('quality')
    binned_X_cols = []
    for col in range(len(X[0])):
        column = [row[col] for row in X]
        binned = myutils.bin_data(column)
        binned_X_cols.append(binned)
    binned_X = []
    for i in range(len(binned_X_cols)):
        row = [item[i] for item in binned_X_cols]
        binned_X.append(row)
    X = binned_X
    return X, y
    # X_train = data.get_columns(['fixed acidity', 'citric acid', 'residual sugar', 'pH', 'sulphates', 'alcohol'])
    # y_train = data.get_column('quality')
    # return X_train, y_train

def main():
    wine_table = MyPyTable()
    filename = os.path.join("data", "WineQT.csv")
    wine_table.load_from_file(filename)

    X_train, y_train = get_training_sets(wine_table)

    wine_knn = MyKNeighborsClassifier(n_neighbors=5)
    wine_knn.fit(X_train, y_train)

    outfile = open("tree.p", "wb")
    pickle.dump(wine_knn, outfile)
    outfile.close()

if __name__ == "__main__":
    main()