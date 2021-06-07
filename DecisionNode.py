# encoding: utf-8

class DecisionNode:

    def __init__(self, feature=-1, threshold=None, label=None, true_branch=None,
                 false_branch=None):
        self.feature = feature
        self.threshold = threshold
        self.label = label  # If label is not None, it's a leaf node.
        self.true_branch = true_branch
        self.false_branch = false_branch
