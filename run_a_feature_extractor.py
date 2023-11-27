


__author__ = 'Stefano Mauceri'
__email__ = 'mauceri.stefano@gmail.com'


"""
In this file I show you how to apply a given
feature-extractor to your data.
The resulting feature-based representation can be used
for classification. If you want to use multiple
features you need to use multiple feature-extractors
separately and concatenate resulting features.
"""



# =============================================================================
# IMPORT
# =============================================================================



import os
import numpy as np
from src.fitness.math_functions import *
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler



# =============================================================================
# FUNCTIONS
# =============================================================================



class KNN(NearestNeighbors):

    k = 1
    metric_params = {}

    def __init__(self, k=1, metric_params={}):
        super().__init__(n_neighbors=k, metric_params=metric_params)



    def score_samples(self, X_test):
        try:
            neighbors = self.kneighbors(X_test)[0]
        except:
            self.algorithm = 'brute'
            self.fit(self._fit_X)
            neighbors = self.kneighbors(X_test)[0]
        return np.mean(neighbors, axis=1) * -1



def extract_features(X, phenotype):
    T = X
    return eval(phenotype)



def scale_features(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)



def get_auroc(Y_true, Test_scores):
    fpr, tpr, _ = roc_curve(Y_true, Test_scores, pos_label=1)
    return auc(fpr, tpr)



# =============================================================================
# MAIN
# =============================================================================



# SET THE DATASET NAME, THE POSITIVE CLASS LABEL, THE FEATURE EXTRACTOR.
dataset = 'SyntheticControl'

class_ = 1

feature_extractor = 'extract(T, 58, None, True, function = lambda T:ARCoeff(T))'


# LOAD DATA.
cwd = os.getcwd()

X_train = np.load(os.path.join(cwd, 'data', f'{dataset}', f'{dataset}_X_TRAIN.npy'))
Y_train = np.load(os.path.join(cwd, 'data', f'{dataset}', f'{dataset}_Y_TRAIN.npy'))

X_test = np.load(os.path.join(cwd, 'data', f'{dataset}', f'{dataset}_X_TEST.npy'))
Y_test = np.load(os.path.join(cwd, 'data', f'{dataset}', f'{dataset}_Y_TEST.npy'))


# ADAPT DATA TO ONE-CLASS CLASSIFICATION.
X_train = X_train[Y_train == class_]
Y_test = (Y_test == class_).astype(int)


# EXTRACT FEATURES.
X_train = extract_features(X_train, feature_extractor)
X_test = extract_features(X_test, feature_extractor)


# STANDARDISE FEATURES.
X_train, X_test = scale_features(X_train, X_test)


# CLASSIFY.
classifier = KNN()
classifier.fit(X_train)
scores = classifier.score_samples(X_test)
auroc = get_auroc(Y_test, scores) * 100

print(f'THE AUROC IS: {round(auroc, 1)}%')



# =============================================================================
# END
# =============================================================================


