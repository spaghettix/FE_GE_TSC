


__author__ = 'Stefano Mauceri'
__email__ = 'mauceri.stefano@gmail.com'



# =============================================================================
# IMPORT
# =============================================================================



import os
import numpy as np
from math_functions import *
from algorithm.parameters import params
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import warnings # to silence sklearn warnings in scale_features_mean_var()
warnings.simplefilter("ignore")



# =============================================================================
# CLASSIFIERS
# =============================================================================



class KNN(NearestNeighbors):



    def __init__(self, k=1, metric_params={}):
        super(KNN, self).__init__(n_neighbors=k, metric_params=metric_params)



    def score_samples(self, X_test):
        try:
            neighbors = self.kneighbors(X_test)[0]
        except:
            self.algorithm = 'brute'
            self.fit(self._fit_X)
            neighbors = self.kneighbors(X_test)[0]
        return np.mean(neighbors, axis=1) * -1



# =============================================================================
# TIME SERIES CLASSIFICATION FITNESS CLASS
# =============================================================================



class tsc_fitness(object):



    def __init__(self):



        # SOME PARAMETERS
        self.maximise = False
        self.default_fitness = np.nan
        self.default_auroc = np.nan
        self.default_Pearson_corr = np.nan
        self.run = params['RUN_NUMBER']
        self.generations = params['GENERATIONS']
        self.feature_number = params['FEATURE_NUMBER']
        self.save_dir = params['SAVE_DIR']
        self.class_ = params['CLASS']
        self.dataset_name = params['DATASET_NAME']
        self.data_dir = params['DATA_DIR']
        self.kfolds = params['KFOLDS']
        current_save_dir = params['FILE_PATH'].split(os.path.sep)[:-2]
        self.current_save_dir = os.path.sep.join(current_save_dir)

        # CLASSIFIER
        self.classifier = KNN()

        # LOAD DATA
        self.X_class_c, self.X_class_not_c, self.split_indices_and_labels = self.load_data()
        self.previous_features = self.load_previous_features()



    def __call__(self, ind):
        return self.get_fitness(ind.phenotype)



    def get_fitness(self, phenotype):

        try:
            x_train = self.extract_features(self.X_class_c, phenotype)
            x_validation = self.extract_features(self.X_class_not_c, phenotype)

            auroc = 0
            for (ix_train, ix_validation, y) in self.split_indices_and_labels:
                train = x_train[ix_train]
                test = np.r_[x_validation, x_train[ix_validation]]
                train, test = self.scale_features(train, test)
                self.classifier.fit(train)
                scores = self.classifier.score_samples(test)
                auroc += self.get_auroc(y, scores)
            auroc /= self.kfolds
            auroc_score = 1 - auroc

            if self.feature_number == 1:
                return auroc_score, self.default_auroc, self.default_Pearson_corr

            else:
                corr = np.corrcoef(np.c_[x_train, self.previous_features])[1:, 0]
                corr = np.mean(np.square(corr))
                return auroc_score + corr, auroc_score, corr

        except:
            return self.default_fitness, self.default_auroc, self.default_Pearson_corr



    def extract_features(self, X, phenotype):
        T = X
        return eval(phenotype)



    def scale_features(self, X_train, X_test):
        scaler = StandardScaler()
        scaler.fit(X_train)
        return scaler.transform(X_train), scaler.transform(X_test)



    def get_auroc(self, Y_true, Test_scores):
        fpr, tpr, _ = roc_curve(Y_true, Test_scores, pos_label=1)
        return auc(fpr, tpr)



    def load_data(self):

        path = os.path.join(self.data_dir, self.dataset_name)
        X = np.load(os.path.join(path, f'{self.dataset_name}_X_TRAIN.npy'))
        Y = np.load(os.path.join(path, f'{self.dataset_name}_Y_TRAIN.npy'))

        X_class_c = X[(Y == self.class_)]
        X_class_not_c = X[(Y != self.class_)]
        Y_not_class_c = np.zeros(X_class_not_c.shape[0])

        nsamples = X_class_c.shape[0]
        train_size = np.ceil(0.66 * nsamples).astype(int)

        split_indices_and_labels = []

        for i in range(self.kfolds):

            ix = np.arange(nsamples)
            ix_train = np.random.choice(ix, size=train_size, replace=False)
            ix_validation = np.setdiff1d(ix, ix_train, assume_unique=True)
            y_validation = np.r_[Y_not_class_c, np.ones_like(ix_validation)]
            split_indices_and_labels.append((ix_train, ix_validation, y_validation))

        return X_class_c, X_class_not_c, split_indices_and_labels



    def load_previous_features(self):

        if self.feature_number == 1:
            return None

        else:

            path = os.path.join(self.current_save_dir, 'FEATURES')
            if not os.path.isdir(path):
                os.mkdir(path)

            path = os.path.join(self.current_save_dir, 'PHENOTYPES')
            if not os.path.isdir(path):
                os.mkdir(path)

            save_phen_path = os.path.join(self.current_save_dir, 'PHENOTYPES', f'CLASS={self.class_}_RUN={self.run}.txt')
            if not os.path.exists(save_phen_path):
                open(save_phen_path, 'a').close()


            path = os.path.join(self.data_dir, self.dataset_name)
            X = np.load(os.path.join(path, f'{self.dataset_name}_X_TRAIN.npy'))
            Y = np.load(os.path.join(path, f'{self.dataset_name}_Y_TRAIN.npy'))
            X_class_c = X[(Y == self.class_)]


            DATA = []
            for fts in range(1, self.feature_number):
                phen_path = os.path.join(self.current_save_dir,
                                         f'F_{fts}',
                                         f'CLASS={self.class_}_RUN={self.run}',
                                         f'{self.generations}.txt')
                fts_path = os.path.join(self.current_save_dir,
                                        'FEATURES',
                                        f'CLASS={self.class_}_RUN={self.run}_F_{fts}.npy')
                try:
                    x = np.load(fts_path)
                    DATA.append(x)
                except:
                    phenotype_file = open(phen_path, 'r')
                    for line in phenotype_file:
                        if line.startswith('Phenotype:'):
                            phenotype = next(phenotype_file).strip()
                            x = self.extract_features(X_class_c, phenotype)
                            np.save(fts_path, x)
                            DATA.append(x)
                    phenotype_file.close()

                    save_phen_file = open(save_phen_path, 'a')
                    save_phen_file.write(phenotype + '\n')
                    save_phen_file.close()

        return np.concatenate(DATA, axis=1)



# =============================================================================
# END
# =============================================================================


