# encoding: utf-8

import numpy as np
import pandas as pd
import time


def cal_label_dic(label_col):
    # For classfication, return a dic, key is label, value is count of label
    dic = {}
    for label in label_col:
        if label not in dic:
            dic[label] = 0
        dic[label] += 1
    return dic


def cal_gini(label_column):
    # For classfication, calculate the gini index.
    total = len(label_column)
    label_dic = cal_label_dic(label_column)
    imp = 0
    for k1 in label_dic:
        p1 = float(label_dic[k1]) / total
        for k2 in label_dic:
            if k1 == k2: continue
            p2 = float(label_dic[k2]) / total
            imp += p1 * p2
    return imp


def voting(label_dic, b=None):
    # For classfication, return majority label.
    if b == None:
        winner_key = list(label_dic.keys())[0]
        for key in label_dic:
            if label_dic[key] > label_dic[winner_key]:
                winner_key = key
            elif label_dic[key] == label_dic[winner_key]:
                winner_key = np.random.choice([key, winner_key], 1)[0]  # return a list with len 1
    else:
        arr = np.array(list(label_dic.items()))
        prob = np.exp(arr[:, 1] * b) / np.exp(arr[:, 1] * b).sum()
        winner_key = np.random.choice(arr[:, 0], size=1, p=prob)[0]

    return winner_key


def load_data():
    # Load the data "car" for the demo
    # Data description: 1728 samples, 6 features and 3 classes

    data = np.array(pd.read_csv('data/car.data', sep=',', header=None, skiprows=0))
    features_attr = ['d', 'd', 'd', 'd', 'd', 'd']

    # data preprocess
    for i in range(data.shape[1]):
        if isinstance(data[0, i], str):
            col = data[:, i]
            new_col = []
            for k in range(len(col)):
                if col[k] is np.nan:
                    data[k, i] = -1
                else:
                    new_col.append(col[k])
            unique_val = np.unique(new_col)
            for num in range(len(unique_val)):
                for k in range(data.shape[0]):
                    if data[k, i] == unique_val[num]:
                        data[k, i] = num

    return data, features_attr


def max_min_normalization(arr):
    # Normalization the information gain
    min_ = np.min(arr)
    max_ = np.max(arr)
    if max_ - min_ == 0:
        return np.zeros(np.shape(arr))
    return (arr - min_) / (max_-min_)


def output_time(flag):
    # Show the time with the flag.
    print(flag, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def mutinomial(norm_seq):
    prob = np.exp(norm_seq) / sum(np.exp(norm_seq))
    np.random.seed()
    random_num = np.random.rand(1)[0]
    last_prob = 0.0
    chosen_pair = 0
    for i in range(len(prob)):
        if last_prob <= random_num < last_prob + prob[i]:
            chosen_pair = i
            break
        last_prob = last_prob + prob[i]
    return int(chosen_pair)
