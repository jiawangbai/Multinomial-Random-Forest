# encoding: utf-8

from utils import *
from DecisionNode import *
import numpy as np
from sklearn.model_selection import train_test_split


class Tree:

    def __init__(self, min_samples_leaf=5, partition_rate=1, B1=5, B2=5, B3=None):
        self.root = None
        self.min_samples_leaf = min_samples_leaf
        self.features_attr = None
        self.b_1 = B1/2                            # Factor that influence selection randomness of splitting feature.
        self.b_2 = B2/2                            # Factor that influence selection randomness of splitting point.
        self.b_3 = B3/2 if B3 != None else B3      # Factor that influence labels in leaf nodes.
        self.partition_rate = partition_rate
        self.criterion = cal_gini

        assert self.b_1 >= 0
        assert self.b_2 >= 0
        assert self.b_3 == None or self.b_3 >= 0

    def fit(self, X, y, features_attr=None):

        # feature_atrr: an array, its size is same as the number of features,
        # 'd': discrete, 'c': continuous
        assert len(features_attr) == X.shape[1]

        np.random.seed()

        self.features_attr = features_attr

        X_e, X_s, y_e, y_s = train_test_split(X, y, test_size=self.partition_rate/(self.partition_rate+1))
        structure_points = np.concatenate((np.array(X_s), np.array([y_s]).T), axis=1)
        estimation_points = np.concatenate((np.array(X_e), np.array([y_e]).T), axis=1)

        self.root = self.__build_tree(structure_points, estimation_points)

    def predict(self, X):

        if np.ndim(X) == 1:
            return self.__predict_rec(X, self.root)
        else:
            result = []
            for sample in X:
                result.append(self.__predict_rec(sample, self.root))
            return np.array(result)

    def __predict_rec(self, X, node):
        if node.label is not None:
            return node.label
        else:
            feat_value = X[node.feature]
            feat_attr = self.features_attr[node.feature]
            threshold = node.threshold

            if feat_value is None or feat_value is np.nan:
                choice = np.random.randint(1, 3)
                if choice == 1:
                    return self.__predict_rec(X, node.true_branch)
                else:
                    return self.__predict_rec(X, node.false_branch)
            else:
                if feat_attr == 'd':
                    if feat_value == threshold:
                        return self.__predict_rec(X, node.true_branch)
                    else:
                        return self.__predict_rec(X, node.false_branch)
                elif feat_attr == 'c':
                    if feat_value >= threshold:
                        return self.__predict_rec(X, node.true_branch)
                    else:
                        return self.__predict_rec(X, node.false_branch)

    def __split(self, dataset, split_feature, threshold):

        true_index = []
        false_index = []

        if self.features_attr[split_feature] == 'd':
            for i in range(len(dataset)):
                if dataset[i][split_feature] == threshold:
                    true_index.append(i)
                else:
                    false_index.append(i)
        elif self.features_attr[split_feature] == 'c':
            for i in range(len(dataset)):
                if dataset[i][split_feature] >= threshold:
                    true_index.append(i)
                else:
                    false_index.append(i)

        return true_index, false_index

    def __split_pair(self, dataset, candidate_features):

        current = self.criterion(dataset[:, -1])

        ret = {}

        for feat in candidate_features:
            col = dataset[:, feat]
            unique_col = np.unique(col)
            attr = self.features_attr[feat]
            ret[feat] = []

            threshold_list = []
            if attr == 'd' or unique_col.shape == 1:
                threshold_list = unique_col
            elif attr == 'c':
                threshold_list = [(unique_col[i]+unique_col[i+1]) / 2 for i in range(len(unique_col)-1)]

            for t in threshold_list:
                true_index, false_index = self.__split(dataset, feat, t)

                p = float(len(true_index)) / len(dataset)
                gain = current - p * self.criterion(dataset[true_index, -1]) - \
                       (1-p) * self.criterion(dataset[false_index, -1])

                ret[feat].append([gain, t])
            ret[feat] = np.array(ret[feat])
            ret[feat] = ret[feat][np.argsort(-ret[feat][:, 0])]

        return ret

    def __build_tree(self, structure_points, estimation_points):

        if len(cal_label_dic(structure_points[:, -1])) == 1:
            return DecisionNode(label=voting(cal_label_dic(estimation_points[:, -1])))

        candidate_features = []
        for i in range(structure_points.shape[1]-1):
            if len(np.unique(structure_points[:, i])) > 1:
                candidate_features.append(i)
        if candidate_features == []:
            return DecisionNode(label=voting(cal_label_dic(estimation_points[:, -1])))

        info_gain_dict = self.__split_pair(structure_points, candidate_features)

        info_gain_feat_max = []
        for key, val in info_gain_dict.items():
            info_gain_feat_max.append([key, val[0][0]])
        info_gain_feat_max = np.array(info_gain_feat_max)
        info_gain_feat_max = info_gain_feat_max[np.argsort(-info_gain_feat_max[:, 1])]

        info_gain_feat_max_norm = self.b_1 * max_min_normalization(info_gain_feat_max[:, 1])
        split_feature = int(info_gain_feat_max[mutinomial(info_gain_feat_max_norm)][0])

        info_gain_chosen_feat_norm = self.b_2 * max_min_normalization(info_gain_dict[split_feature][:, 0])
        threshold = info_gain_dict[split_feature][mutinomial(info_gain_chosen_feat_norm)][1]

        true_index_s, false_index_s = self.__split(structure_points, split_feature, threshold)
        true_index_e, false_index_e = self.__split(estimation_points, split_feature, threshold)

        if len(true_index_e) == 0 or len(false_index_e) == 0:
            return DecisionNode(label=voting(cal_label_dic(estimation_points[:, -1]), self.b_3))

        if len(true_index_e) <= self.min_samples_leaf:
            true_branch = DecisionNode(label=voting(cal_label_dic(estimation_points[true_index_e, -1])))
        else:
            true_branch = self.__build_tree(structure_points[true_index_s], estimation_points[true_index_e])

        if len(false_index_e) <= self.min_samples_leaf:
            false_branch = DecisionNode(label=voting(cal_label_dic(estimation_points[false_index_e, -1])))
        else:
            false_branch = self.__build_tree(structure_points[false_index_s], estimation_points[false_index_e])

        return DecisionNode(feature=split_feature, threshold=threshold,
                            true_branch=true_branch, false_branch=false_branch)
