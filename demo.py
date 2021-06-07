# encoding: utf-8

import warnings
warnings.filterwarnings('ignore')
import imp
from MultinomialRF import MultinomialRF
import numpy as np
from sklearn.model_selection import KFold
from utils import load_data, output_time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CROSS_VALIDATION = False   # run as cross validation or not
ROUND_NUM = 10            # times of cross validation
FOLD_NUM = 10             # fold number of cross validation


def train_test(train_set, test_set, feature_attr):
    # Train and test respectively.

    # Create Multinomial Random Forest, which is similar to use scikit-learn's API.
    clf = MultinomialRF(n_estimators=100,
                        min_samples_leaf=5,
                        B1=5,
                        B2=5,
                        B3=None,
                        partition_rate=1,
                        n_jobs=1)

    clf.fit(train_set[:, :-1], train_set[:, -1], feature_attr)

    pred = clf.predict(test_set[:, :-1])

    return accuracy_score(pred, test_set[:, -1].astype(int))


def cross_validation(data, feature_attr):
    # Cross validation with the input data.
    print("{0}-fold cross validation with {1} times.".format(str(FOLD_NUM), str(ROUND_NUM)))

    res = []
    for i in range(ROUND_NUM):
        kf = KFold(n_splits=FOLD_NUM, shuffle=True)
        for train_index, test_index in kf.split(X=data[:, :-1], y=data[:, -1], groups=data[:, -1]):
            train_set, test_set = data[train_index], data[test_index]
            res.append(train_test(train_set, test_set, feature_attr))

        print("ROUND[{0}] accuracy score: {1}".format(str(i + 1), str(np.mean(res[-FOLD_NUM:]))[:6]))

    return np.mean(res)


if __name__ == "__main__":

    output_time("START")

    data, feature_attr = load_data()
    np.random.shuffle(data)
    print("Data Size:", data.shape)

    if CROSS_VALIDATION:
        res = cross_validation(data, feature_attr)
    else:
        train_set, test_set = train_test_split(data, test_size=0.1, shuffle=True)
        res = train_test(train_set, test_set, feature_attr)

    print("FINAL accuracy score: {0}".format(str(res)[:6]))

    output_time("END")
