# encoding: utf-8

from utils import *
from Tree import Tree
import numpy as np
from multiprocessing import Pool


class MultinomialRF:

    def __init__(self, n_estimators=100, min_samples_leaf=5,
                 B1=10, B2=10, B3=None, partition_rate=1, n_jobs=4):
        # Below is the parameters and brief introduction, reference the paper for more details.

        self.n_estimators = n_estimators           # Number of trees in the forest.
        self.min_samples_leaf = min_samples_leaf   # Minimum number of samples required to be at a leaf node.
        self.b_1 = B1
        self.b_2 = B2
        self.b_3 = B3
        self.partition_rate = partition_rate       # Partition_rate is |Structure points|/|Estimation points|.
        self.n_jobs = n_jobs                       # Number of jobs to run in parallel for training.
        self.tree_list = None                      # All trees of this forest.
        self.features_attr = None                  # Attributes of features, 'd' 'is discrete', 'c' is continuous

        assert self.b_1 >= 0
        assert self.b_2 >= 0
        assert self.b_3 == None or self.b_3 >= 0

    def fit(self, X, y, features_attr=None):
        # Train with the features X and labels y

        assert len(features_attr) == X.shape[1]
        self.features_attr = features_attr

        self.tree_list = np.array([])

        pool = Pool(processes=self.n_jobs)
        jobs_set = []
        for i in range(self.n_estimators):
            tree = Tree(min_samples_leaf=self.min_samples_leaf,
                        B1=self.b_1,
                        B2=self.b_2,
                        B3=self.b_3,
                        partition_rate=self.partition_rate)
            jobs_set.append(pool.apply_async(self.train_one_tree,
                                             (tree, X, y, self.features_attr, )))
        pool.close()
        pool.join()

        for job in jobs_set:
            self.tree_list = np.append(self.tree_list, job.get())

    def predict(self, X):
        # Predict with feature(s) X, one or more samples.
        tree_pred_res = []
        for tree in self.tree_list:
            tree_pred_res.extend([tree.predict(X)])
        tree_pred_res = np.array(tree_pred_res).T

        if np.ndim(tree_pred_res) == 1:
            return voting(cal_label_dic(tree_pred_res))
        else:
            return np.array([voting(cal_label_dic(res))
                             for res in tree_pred_res])

    @staticmethod
    def train_one_tree(tree, X_train, y_train, features_attr):
        # Train a tree in a thread
        tree.fit(X_train, y_train, features_attr)

        return tree