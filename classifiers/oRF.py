import numpy as np
import sklearn
import copy
from sklearn.base import clone

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class ObliqueRandomForest(BaseEstimator, ClassifierMixin):
    __name__ = 'Oblique Random Forest'

    def __init__(self, n_trees=10, n_fs=250, max_depth=3, node_classifier=None):
        self.n_trees = n_trees
        self.n_fs = n_fs
        self.node_classifier = node_classifier
        if self.node_classifier == None:
            self.node_classifier = LinearSVC(C=0.1)
        self.max_depth = max_depth
        self.rf = []
        self.th = 0

    def fit(self, X, y):

        for i in range(0, self.n_trees):
            classifier = clone(self.node_classifier)
            tree = ObliqueTree(n_fs=self.n_fs, max_depth=self.max_depth, node_classifier=classifier)
            tree.fit(X, y)
            self.rf.append(tree)

        y_pred = self.decision_function(X)
        self.th = self.gini_index(y_predict=y_pred, y_expected=y)

        return self

    def gini_index(self, y_predict, y_expected):
        gini = []
        c1 = np.where(y_expected != 1)  # Negative samples
        c2 = np.where(y_expected == 1)  # Positive samples
        n = y_expected.shape[0]
        thresholds = y_predict
        for th in thresholds:

            tmp = np.where(y_predict[c1] < th)[0]  # Predict correctly the negative sample
            c1c1 = tmp.shape[0]
            n1 = c1c1

            tmp = np.where(y_predict[c1] >= th)[0]  # Predict the negative sample as positive
            c1c2 = tmp.shape[0]
            n2 = c1c2

            tmp = np.where(y_predict[c2] >= th)[0]  # Predict correctly the positive sample
            c2c2 = tmp.shape[0]
            n2 = n2 + c2c2
            tmp = np.where(y_predict[c2] < th)[0]  # Predict the positive samples as negative
            c2c1 = tmp.shape[0]
            n1 = n1 + c2c1

            if n1 == 0 or n2 == 0:
                gini.append(9999)
                continue
            else:
                gini1 = (c1c1 / n1) ** 2 - (c2c1 / n1) ** 2
                gini1 = 1 - gini1
                gini1 = (n1 / n) * gini1

                gini2 = (c2c2 / n2) ** 2 - (c1c2 / n2) ** 2
                gini2 = 1 - gini2
                gini2 = (n2 / n) * gini2

                gini.append(gini1 + gini2)
        if len(gini) > 0:
            idx = gini.index(min(gini))
            best_th = thresholds[idx]
        else:
            print('Threshold not found')
            best_th = 0

        return best_th

    def predict(self, X):
        scores = self.decision_function(X)
        for i in range(0, scores.shape[0]):
            if scores[i] < self.th:
                scores[i] = 0
            else:
                scores[i] = 1
        return scores

    def decision_function(self, X):
        n_samples = X.shape[0]
        scores = np.zeros((n_samples,1))
        for sample_idx in range(0, n_samples):
            for i in range(0, self.n_trees):
                sample = np.expand_dims(X[sample_idx], axis=0)
                scores[sample_idx] += self.rf[i].decision_function(sample)

            scores[sample_idx] = scores[sample_idx]/len(self.rf)
        return scores

class ObliqueTree(BaseEstimator, ClassifierMixin):
    __name__ = 'Oblique Decision Tree'

    def __init__(self, n_fs=250, max_depth=3, node_classifier=None, depth=0):
        self.n_fs = n_fs
        self.node_classifier = node_classifier
        self.max_depth = max_depth
        self.l_child = None
        self.r_child = None
        self.model = None
        self.depth = depth
        self.th=0
        self.idxs = None

    def fit(self, X, y):
        if self.depth < self.max_depth:
            self.idxs = np.random.choice(X.shape[1], self.n_fs)
            self.model = self.node_classifier.fit(X[:,self.idxs], y)
            pred = self.model.decision_function(X[:,self.idxs])
            self.th = self.gini_index(y_predict=pred, y_expected=y)

            samples_left = np.where(pred < self.th)[0]
            samples_right = np.where(pred >= self.th)[0]

            if samples_left.shape[0] > 0:
                c1 = np.where(y[samples_left]==1)[0].shape[0]
                c2 = y[samples_left].shape[0]-c1

                if c1!=0 and c2!= 0:
                    left_child = ObliqueTree(n_fs=self.n_fs, max_depth=self.max_depth,
                                             node_classifier=self.node_classifier, depth=self.depth+1)
                    left_child.fit(X[samples_left], y[samples_left])
                    self.l_child = left_child

            if samples_right.shape[0] >0:
                c1 = np.where(y[samples_right] == 1)[0].shape[0]
                c2 = y[samples_right].shape[0] - c1

                if c1 != 0 and c2 != 0:
                    right_child = ObliqueTree(n_fs=self.n_fs, max_depth=self.max_depth,
                                              node_classifier=self.node_classifier, depth=self.depth+1)
                    right_child.fit(X[samples_right], y[samples_right])
                    self.r_child = right_child

        return self

    def decision_function(self, X):
        root = self
        final_score = 0
        level = 1
        while root is not None and root.model is not None:
            score = root.model.decision_function(X[:, root.idxs])
            final_score+=score
            level += 1

            if score < root.th:
                root = root.l_child
            else:
                root = root.r_child

        return final_score/level

    def gini_index(self, y_predict, y_expected):
        gini = []
        c1 = np.where(y_expected != 1)  # Negative samples
        c2 = np.where(y_expected == 1)  # Positive samples
        n = y_expected.shape[0]
        thresholds = y_predict
        for th in thresholds:

            tmp = np.where(y_predict[c1] < th)[0]  # Predict correctly the negative sample
            c1c1 = tmp.shape[0]
            n1 = c1c1

            tmp = np.where(y_predict[c1] >= th)[0]  # Predict the negative sample as positive
            c1c2 = tmp.shape[0]
            n2 = c1c2

            tmp = np.where(y_predict[c2] >= th)[0]  # Predict correctly the positive sample
            c2c2 = tmp.shape[0]
            n2 = n2 + c2c2
            tmp = np.where(y_predict[c2] < th)[0]  # Predict the positive samples as negative
            c2c1 = tmp.shape[0]
            n1 = n1 + c2c1

            if n1 == 0 or n2 == 0:
                gini.append(9999)
                continue
            else:
                gini1 = (c1c1 / n1) ** 2 - (c2c1 / n1) ** 2
                gini1 = 1 - gini1
                gini1 = (n1 / n) * gini1

                gini2 = (c2c2 / n2) ** 2 - (c1c2 / n2) ** 2
                gini2 = 1 - gini2
                gini2 = (n2 / n) * gini2

                gini.append(gini1 + gini2)
        if len(gini) > 0:
            idx = gini.index(min(gini))
            best_th = thresholds[idx]
        else:
            print('Threshold not found')
            best_th = 0

        return best_th

if __name__ == '__main__':
    np.random.seed(12227)
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    import pls_classifier as pls
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    import pls_classifier as pls

    X, y = make_classification(n_samples=1000, n_features=160, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #clf = ObliqueRandomForest(n_trees=40, n_fs=80, node_classifier=pls.PLSClassifier(n_comp=2))
    #clf = ObliqueRandomForest(n_trees=40, n_fs=80, node_classifier=LinearSVC(C=0.1))

    pls_model = pls.PLSClassifier(n_comp=10, estimator=QuadraticDiscriminantAnalysis())
    clf = ObliqueRandomForest(n_trees=2, n_fs=80, node_classifier=pls_model)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred, normalize=True))

    y_pred = clf.decision_function(X_test)
    print(roc_auc_score(y_test, y_pred))
