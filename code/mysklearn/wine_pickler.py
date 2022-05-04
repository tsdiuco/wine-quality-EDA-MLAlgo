from myclassifiers import MyRandomForestClassifier
from mypytable import MyPyTable

import pickle
import os

def get_training_sets(data):
    X_train = [row[:-2] for row in data]
    y_train = [row[-2] for row in data]
    return X_train, y_train

def main():
    wine_table = MyPyTable()
    filename = os.path.join("../data", "WineQT.csv")
    wine_table.load_from_file(filename)

    X_train, y_train = get_training_sets(wine_table.data)
    print(X_train)
    print(y_train)

    wine_random_forest = MyRandomForestClassifier(15, 5, 4)
    wine_random_forest.fit(X_train, y_train)
    print(wine_random_forest.forest)

    outfile = open("tree.p", "wb")
    pickle.dump(wine_random_forest.forest, outfile)
    outfile.close()

if __name__ == "__main__":
    main()