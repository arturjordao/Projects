import numpy as np
import sklearn
import copy
from sklearn.base import clone

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression


class VIPClassifier(BaseEstimator, ClassifierMixin):
    __name__ = 'VIP Classifier'

    def __init__(self, n_comp=2, estimator=LinearRegression(), percentage_discard=0.1, feat_idx=[]):
        self.n_comp = n_comp
        self.percentage_discard = percentage_discard
        self.feat_idx = feat_idx
        self.estimator = estimator

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

    def vip(self, x, model):
        #Adapted from https://github.com/scikit-learn/scikit-learn/issues/7050
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_

        m, p = x.shape
        _, h = t.shape

        vips = np.zeros((p,))

        # s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        s = np.diag(np.dot(np.dot(np.dot(t.T, t), q.T), q)).reshape(h, -1)
        total_s = np.sum(s)

        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
            # vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
            vips[i] = np.sqrt(p * (np.dot(s.T, weight)) / total_s)

        return vips

    def th_to_discard(self, scores, percentage=0.1):
        total = scores.shape[0]
        closest = np.zeros(total)
        for i in range(0, total):
            th = scores[i]
            idxs = np.where(scores <= th)[0]
            discarded = len(idxs) / total
            closest[i] = abs(percentage - discarded)

        th = scores[np.argmin(closest)]
        return th

    def fit(self, X, y):
        dm = PLSRegression(n_components=self.n_comp, scale=True)
        dm.fit(X, y)

        scores = self.vip(X, dm)
        th = self.th_to_discard(scores, percentage=self.percentage_discard)
        self.feat_idx = np.where(scores >= th)[0]
        X = X[:,self.feat_idx]

        self.estimator.fit(X, y)
        y_pred = self.estimator.decision_function(X)
        self.th = self.gini_index(y_pred, y)
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, 1))

        for i in range(0, n_samples):
            scores[i] = self.decision_function(X[i])

        aux = copy.deepcopy(scores)
        aux[scores < self.th] = 0
        aux[scores >= self.th] = 1
        scores = aux

        return scores

    def decision_function(self, X):
        #Predict multiple samples
        if len(X.shape) == 2:
            X = X[:,self.feat_idx]
        #Predict single sample
        else:
            X = X[self.feat_idx]
        scores = self.estimator.decision_function(X)

        return scores


if __name__ == '__main__':
    np.random.seed(12227)
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


    X, y = make_classification(n_samples=1000, n_features=160, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = VIPClassifier(n_comp=2, percentage_discard=0.1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred, normalize=True))

    y_pred = clf.decision_function(X_test)
    print(roc_auc_score(y_test, y_pred))
