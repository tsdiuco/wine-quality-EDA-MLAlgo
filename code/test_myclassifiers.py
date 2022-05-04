import numpy as np

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier,\
    MyRandomForestClassifier,\
    MyDecisionTreeClassifier,\
    MyNaiveBayesClassifier

# note: order is actual/received student value, expected/solution
def test_simple_linear_regression_classifier_fit():
    np.random.seed(0)
    # y = 2x + noise
    X_train = [[val] for val in range(100)]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]
    discretizer = lambda y: "high" if y <= 100 else "low"
    classifier = MySimpleLinearRegressionClassifier(discretizer, regressor=MySimpleLinearRegressor())
    classifier.fit(X_train, y_train)
    assert np.isclose(classifier.regressor.slope, 1.9, atol=0.1)
    assert np.isclose(classifier.regressor.intercept, 5.2, atol=0.1)
    # y = 3x + 5 + noise
    X_train = [[val] for val in range(100)]
    y_train = [row[0] * 3 + 5 + np.random.normal(0, 25) for row in X_train]
    discretizer = lambda y: "high" if y <= 80 else "low"
    classifier = MySimpleLinearRegressionClassifier(discretizer, regressor=MySimpleLinearRegressor())
    classifier.fit(X_train, y_train)
    assert np.isclose(classifier.regressor.slope, 2.854, atol=0.01)
    assert np.isclose(classifier.regressor.intercept, 14.2761, atol=0.01)


def test_simple_linear_regression_classifier_predict():
    X_test = [[i] for i in range(200)]
    # y = 0.5x - 3 + noise
    X_train = [[val] for val in range(100)]
    y_train = [row[0] * 0.5 - 3 + np.random.normal(0, 25) for row in X_train]
    discretizer = lambda y: "high" if y <= 100 else "low"
    classifier = MySimpleLinearRegressionClassifier(discretizer, regressor=MySimpleLinearRegressor())
    classifier.fit(X_train, y_train)
    y_predicted = classifier.predict(X_test)
    # y = 2x + 1 + noise
    X_train = [[val] for val in range(100)]
    y_train = [row[0] * 2 + 1 + np.random.normal(0, 25) for row in X_train]
    discretizer = lambda y: "high" if y <= 80 else "low"
    classifier = MySimpleLinearRegressionClassifier(discretizer, regressor=MySimpleLinearRegressor())
    classifier.fit(X_train, y_train)
    y_predicted = classifier.predict(X_test)

def test_kneighbors_classifier_kneighbors():
    test_instance = [[2, 3]]
    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    classifier = MyKNeighborsClassifier()
    classifier.fit(X_train_class_example1, y_train_class_example1)
    distances, neighbor_indices = classifier.kneighbors(test_instance)
    assert neighbor_indices == [[[0, 2.23606797749979], [1, 3.1622776601683795], [2, 3.433496759864497], [3, 3.605551275463989]]]
    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    classifier.fit(X_train_class_example2, y_train_class_example2)
    distances, neighbor_indices = classifier.kneighbors(test_instance)
    assert neighbor_indices == [[[0, 1.4142135623730951], [1, 5.0], [2, 2.8284271247461903], [3, 2.23606797749979], [4, 1.4142135623730951], [5, 3.0], [6, 2.0], [7, 3.1622776601683795]]]
    # from Bramer
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]
    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]
    classifier.fit(X_train_bramer_example, y_train_bramer_example)
    distances, neighbor_indices = classifier.kneighbors(test_instance)
    assert neighbor_indices == [[[0, 3.5114099732158874], [1, 5.1351728305871065], [2, 4.401136216933078], [3, 11.315917991926241], [4, 10.73312629199899], [5, 9.616652224137047], [6, 11.21605991424796], [7, 11.0], [8, 11.985407794480755], [9, 10.973149046650192], [10, 10.965856099730654], [11, 20.727035485085654], [12, 19.725364381932213], [13, 19.807069445023913], [14, 16.58553586713435], [15, 15.061540425866141], [16, 15.47287949930458], [17, 16.66283289239858], [18, 17.00470523119998], [19, 19.374467734624353]]]

def test_kneighbors_classifier_predict():
    test_instance = [[2, 3]]
    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    classifier = MyKNeighborsClassifier()
    classifier.fit(X_train_class_example1, y_train_class_example1)
    y_predicted = classifier.predict(test_instance)
    assert y_predicted == ['bad']
    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    classifier.fit(X_train_class_example2, y_train_class_example2)
    y_predicted = classifier.predict(test_instance)
    assert y_predicted == ['no']
    # from Bramer
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]
    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    classifier.fit(X_train_bramer_example, y_train_bramer_example)
    y_predicted = classifier.predict(test_instance)
    assert y_predicted == ['-']

def test_dummy_classifier_fit():
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    classifier = MyDummyClassifier()
    classifier.fit([], y_train)
    assert classifier.most_common_label == "yes"
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    classifier = MyDummyClassifier()
    classifier.fit([], y_train)
    assert classifier.most_common_label == "no"

def test_dummy_classifier_predict():
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    classifier = MyDummyClassifier()
    classifier.fit([], y_train)
    assert classifier.predict([1, 2, 3]) == ["yes", "yes", "yes"]
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    classifier = MyDummyClassifier()
    classifier.fit([], y_train)
    assert classifier.predict([1, 2, 3]) == ["no", "no", "no"]

def test_naive_bayes_classifier_fit():
    # in-class Naive Bayes example (lab task #1)
    inclass_example_col_names = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    classifier = MyNaiveBayesClassifier()
    classifier.fit(X_train_inclass_example, y_train_inclass_example)
    assert classifier.priors == {'yes': 0.625, 'no': 0.375}
    assert classifier.posteriors == {'yes': {'att1': {1: 0.8, 2: 0.2}, 'att2': {5: 0.4, 6: 0.6}}, 'no': {'att1': {1: 2/3, 2: 1/3}, 'att2': {5: 2/3, 6: 1/3}}}

    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]
    X_iphone_train = [x[0:3] for x in iphone_table]
    y_iphone_train = [x[3] for x in iphone_table]
    classifier.fit(X_iphone_train, y_iphone_train)
    assert classifier.priors == {'yes': 2/3, 'no': 1/3}
    assert classifier.posteriors == {'yes': {'att1': {1: 0.2, 2: 0.8}, 'att2': {1: 0.3, 2: 0.4, 3: 0.3}, 'att3': {'fair': 0.7, 'excellent': 0.3}}, 'no': {'att1': {1: 0.6, 2: 0.4}, 'att2': {1: 0.2, 2: 0.4, 3: 0.4}, 'att3': {'fair': 0.4, 'excellent': 0.6}}}

    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]
    X_bramer_train = [x[0:4] for x in train_table]
    y_bramer_train = [x[4] for x in train_table]
    classifier.fit(X_bramer_train, y_bramer_train)
    assert classifier.priors == {'on time': 0.7, 'late': 0.1, 'very late': 0.15, 'cancelled': 0.05}
    assert classifier.posteriors == {'on time': {'att1': {'weekday': 9/14, 'saturday': 1/7, 'holiday': 1/7, 'sunday': 1/14}, 'att2': {'spring': 2/7, 'autumn': 1/7, 'winter': 1/7, 'summer': 3/7}, 'att3': {'normal': 5/14, 'high': 2/7, 'none': 5/14}, 'att4': {'none': 5/14, 'slight': 4/7, 'heavy': 1/14}}, 'late': {'att1': {'weekday': 0.5, 'saturday': 0.5}, 'att2': {'winter': 1.0}, 'att3': {'normal': 0.5, 'high': 0.5}, 'att4': {'none': 0.5, 'heavy': 0.5}}, 'very late': {'att1': {'weekday': 1.0}, 'att2': {'winter': 2/3, 'autumn': 1/3}, 'att3': {'normal': 2/3, 'high': 1/3}, 'att4': {'none': 1/3, 'heavy': 2/3}}, 'cancelled': {'att1': {'saturday': 1.0}, 'att2': {'spring': 1.0}, 'att3': {'high': 1.0}, 'att4': {'heavy': 1.0}}}

def test_naive_bayes_classifier_predict():
    # in-class Naive Bayes example (lab task #1)
    inclass_example_col_names = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    classifier = MyNaiveBayesClassifier()
    classifier.fit(X_train_inclass_example, y_train_inclass_example)
    X_test = [[1, 5]]
    y_pred = classifier.predict(X_test)
    y_desk_calculation = ["yes"]
    assert y_pred == y_desk_calculation

    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]
    X_iphone_train = [x[0:3] for x in iphone_table]
    y_iphone_train = [x[3] for x in iphone_table]
    classifier.fit(X_iphone_train, y_iphone_train)
    X_test = [[2, 2, "fair"], [1, 1, "excellent"]]
    y_pred = classifier.predict(X_test)
    y_desk_calculation = ["yes", "no"]
    assert y_pred == y_desk_calculation

    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]
    X_bramer_train = [x[0:4] for x in train_table]
    y_bramer_train = [x[4] for x in train_table]
    classifier.fit(X_bramer_train, y_bramer_train)
    X_test = [["weekday", "winter", "high", "heavy"]]
    y_pred = classifier.predict(X_test)
    y_desk_calculation = ["very late"]
    assert y_pred == y_desk_calculation

def test_decision_tree_classifier_fit():
    # interview dataset
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    interview_table = [
            ["Senior", "Java", "no", "no", "False"],
            ["Senior", "Java", "no", "yes", "False"],
            ["Mid", "Python", "no", "no", "True"],
            ["Junior", "Python", "no", "no", "True"],
            ["Junior", "R", "yes", "no", "True"],
            ["Junior", "R", "yes", "yes", "False"],
            ["Mid", "R", "yes", "yes", "True"],
            ["Senior", "Python", "no", "no", "False"],
            ["Senior", "R", "yes", "no", "True"],
            ["Junior", "Python", "yes", "no", "True"],
            ["Senior", "Python", "yes", "yes", "True"],
            ["Mid", "Python", "no", "yes", "True"],
            ["Mid", "Java", "yes", "no", "True"],
            ["Junior", "Python", "no", "yes", "False"]
        ]
    X_interview_train = [x[0:4] for x in interview_table]
    y_interview_train = [x[4] for x in interview_table]
    classifier = MyDecisionTreeClassifier()
    classifier.fit(X_interview_train, y_interview_train)

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    interview_tree = \
            ["Attribute", "att0",
                ["Value", "Junior",
                    ["Attribute", "att3",
                        ["Value", "no",
                            ["Leaf", "True", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "False", 2, 5]
                        ]
                    ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Senior",
                    ["Attribute", "att2",
                        ["Value", "no",
                            ["Leaf", "False", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "True", 2, 5]
                        ]
                    ]
                ]
            ]
    assert classifier.tree == interview_tree

    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    degrees_table = [
            ["A", "B", "A", "B", "B", "SECOND"],
            ["A", "B", "B", "B", "A", "FIRST"],
            ["A", "A", "A", "B", "B", "SECOND"],
            ["B", "A", "A", "B", "B", "SECOND"],
            ["A", "A", "B", "B", "A", "FIRST"],
            ["B", "A", "A", "B", "B", "SECOND"],
            ["A", "B", "B", "B", "B", "SECOND"],
            ["A", "B", "B", "B", "B", "SECOND"],
            ["A", "A", "A", "A", "A", "FIRST"],
            ["B", "A", "A", "B", "B", "SECOND"],
            ["B", "A", "A", "B", "B", "SECOND"],
            ["A", "B", "B", "A", "B", "SECOND"],
            ["B", "B", "B", "B", "A", "SECOND"],
            ["A", "A", "B", "A", "B", "FIRST"],
            ["B", "B", "B", "B", "A", "SECOND"],
            ["A", "A", "B", "B", "B", "SECOND"],
            ["B", "B", "B", "B", "B", "SECOND"],
            ["A", "A", "B", "A", "A", "FIRST"],
            ["B", "B", "B", "A", "A", "SECOND"],
            ["B", "B", "A", "A", "B", "SECOND"],
            ["B", "B", "B", "B", "A", "SECOND"],
            ["B", "A", "B", "A", "B", "SECOND"],
            ["A", "B", "B", "B", "A", "FIRST"],
            ["A", "B", "A", "B", "B", "SECOND"],
            ["B", "A", "B", "B", "B", "SECOND"],
            ["A", "B", "B", "B", "B", "SECOND"],
        ]
    X_degrees_train = [x[0:5] for x in degrees_table]
    y_degrees_train = [x[5] for x in degrees_table]
    classifier.fit(X_degrees_train, y_degrees_train)

    degrees_tree = \
        ["Attribute", "att0",
            ["Value", "A",
                ["Attribute", "att4",
                    ["Value", "A",
                        ["Leaf", "FIRST", 5, 14]
                    ],
                    ["Value", "B",
                        ["Attribute", "att3",
                            ["Value", "A",
                                ["Attribute", "att1",
                                    ["Value", "A",
                                        ["Leaf", "FIRST", 1, 2]
                                    ],
                                    ["Value", "B",
                                        ["Leaf", "SECOND", 1, 2]
                                    ]
                                ]
                            ],
                            ["Value", "B",
                                ["Leaf", "SECOND", 7, 9]
                            ]
                        ]
                    ]
                ]
            ],
            ["Value", "B",
                ["Leaf", "SECOND", 12, 26]
            ]
        ]
    assert classifier.tree == degrees_tree

    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]
    X_iphone_train = [x[0:3] for x in iphone_table]
    y_iphone_train = [x[3] for x in iphone_table]
    classifier.fit(X_iphone_train, y_iphone_train)

    iphone_tree = \
        ["Attribute", "att0",
            ["Value", 1,
                ["Attribute", "att1",
                    ["Value", 1,
                        ["Leaf", "yes", 1, 5]
                    ],
                    ["Value", 2,
                        ["Attribute", "att2",
                            ["Value", "excellent",
                                ["Leaf", "yes", 1, 2]
                            ],
                            ["Value", "fair",
                                ["Leaf", "no", 1, 2]
                            ]
                        ]
                    ],
                    ["Value", 3,
                        ["Leaf", "no", 2, 5]
                    ]
                ]
            ],
            ["Value", 2,
                ["Attribute", "att2",
                    ["Value", "excellent",
                        ["Leaf", "no", 2, 4]
                    ],
                    ["Value", "fair",
                        ["Leaf", "yes", 6, 10]
                    ]
                ]
            ]
        ]
    assert classifier.tree == iphone_tree

def test_decision_tree_classifier_predict():
    # interview dataset
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    interview_table = [
            ["Senior", "Java", "no", "no", "False"],
            ["Senior", "Java", "no", "yes", "False"],
            ["Mid", "Python", "no", "no", "True"],
            ["Junior", "Python", "no", "no", "True"],
            ["Junior", "R", "yes", "no", "True"],
            ["Junior", "R", "yes", "yes", "False"],
            ["Mid", "R", "yes", "yes", "True"],
            ["Senior", "Python", "no", "no", "False"],
            ["Senior", "R", "yes", "no", "True"],
            ["Junior", "Python", "yes", "no", "True"],
            ["Senior", "Python", "yes", "yes", "True"],
            ["Mid", "Python", "no", "yes", "True"],
            ["Mid", "Java", "yes", "no", "True"],
            ["Junior", "Python", "no", "yes", "False"]
        ]
    X_interview_train = [x[0:4] for x in interview_table]
    y_interview_train = [x[4] for x in interview_table]
    classifier = MyDecisionTreeClassifier()
    classifier.fit(X_interview_train, y_interview_train)
    X_test = []

    X_test = [["Junior", "Java", "yes", "no"],
        ["Junior", "Java", "yes", "yes"]]
    y_test = ["True", "False"]
    y_pred = classifier.predict(X_test)
    assert y_pred == y_test

    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    degrees_table = [
            ["A", "B", "A", "B", "B", "SECOND"],
            ["A", "B", "B", "B", "A", "FIRST"],
            ["A", "A", "A", "B", "B", "SECOND"],
            ["B", "A", "A", "B", "B", "SECOND"],
            ["A", "A", "B", "B", "A", "FIRST"],
            ["B", "A", "A", "B", "B", "SECOND"],
            ["A", "B", "B", "B", "B", "SECOND"],
            ["A", "B", "B", "B", "B", "SECOND"],
            ["A", "A", "A", "A", "A", "FIRST"],
            ["B", "A", "A", "B", "B", "SECOND"],
            ["B", "A", "A", "B", "B", "SECOND"],
            ["A", "B", "B", "A", "B", "SECOND"],
            ["B", "B", "B", "B", "A", "SECOND"],
            ["A", "A", "B", "A", "B", "FIRST"],
            ["B", "B", "B", "B", "A", "SECOND"],
            ["A", "A", "B", "B", "B", "SECOND"],
            ["B", "B", "B", "B", "B", "SECOND"],
            ["A", "A", "B", "A", "A", "FIRST"],
            ["B", "B", "B", "A", "A", "SECOND"],
            ["B", "B", "A", "A", "B", "SECOND"],
            ["B", "B", "B", "B", "A", "SECOND"],
            ["B", "A", "B", "A", "B", "SECOND"],
            ["A", "B", "B", "B", "A", "FIRST"],
            ["A", "B", "A", "B", "B", "SECOND"],
            ["B", "A", "B", "B", "B", "SECOND"],
            ["A", "B", "B", "B", "B", "SECOND"],
        ]
    X_degrees_train = [x[0:5] for x in degrees_table]
    y_degrees_train = [x[5] for x in degrees_table]
    classifier.fit(X_degrees_train, y_degrees_train)
    X_test = [["B", "B", "B", "B", "B"],
              ["A", "A", "A", "A", "A"],
              ["A", "A", "A", "A", "B"]]
    y_test = ['SECOND', 'FIRST', 'FIRST']
    y_pred = classifier.predict(X_test)
    assert y_pred == y_test

    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]
    X_iphone_train = [x[0:3] for x in iphone_table]
    y_iphone_train = [x[3] for x in iphone_table]
    classifier.fit(X_iphone_train, y_iphone_train)
    X_test = [[2, 2, "fair"], [1, 1, "excellent"]]
    y_test = ['yes', 'yes']
    y_pred = classifier.predict(X_test)
    assert y_pred == y_test

def test_random_forest_classifier_fit():
    # interview dataset
    header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]
    classifier = MyRandomForestClassifier(n_trees=20, m=7, f=2)
    X = [x[0:4] for x in table]
    y = [x[4] for x in table]
    classifier.fit(X, y)

def test_random_forest_classifier_predict():
    # interview dataset
    header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]
    classifier = MyRandomForestClassifier(n_trees=20, m=7, f=2)
    X = [x[0:4] for x in table]
    y = [x[4] for x in table]
    classifier.fit(X, y)
    new_instance = [["Mid", "Python", "yes", "yes"]]
    pred = classifier.predict(new_instance)
    assert pred == ["True"]