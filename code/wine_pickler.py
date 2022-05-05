from mysklearn.myclassifiers import MyKNeighborsClassifier, MyRandomForestClassifier
from mysklearn.mypytable import MyPyTable
import mysklearn.myutils as myutils

import pickle
import os

def get_training_sets(data):
    X_train = data.get_columns(['fixed acidity', 'citric acid', 'residual sugar', 'pH', 'sulphates', 'alcohol'])
    y_train = data.get_column('quality')
    return X_train, y_train

def main():
    wine_random = MyRandomForestClassifier(n_trees=20, m=5, f=3)
    wine_table = MyPyTable()
    filename = os.path.join("data", "WineQT.csv")
    wine_table.load_from_file(filename)

    X_train, y_train = get_training_sets(wine_table)

    wine_random.fit(X_train, y_train)

    outfile = open("tree.p", "wb")
    pickle.dump(wine_random, outfile)
    outfile.close()

if __name__ == "__main__":
    main()