__docformat__ = 'restructedtext en'
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle as skl_shuffle
import numpy as np
from ifclibs import loaders, cleaning, organizing
import pickle

__author__ = "Miquel Perello Nieto"
__credits__ = ["Miquel Perello Nieto"]

__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Miquel Perello Nieto"
__email__ = "miquel@perellonieto.com"
__status__ = "Development"

import urllib
from os.path import isfile

class Dataset(object):
    def __init__(self, name, data, target, test_fold, nested_test_folds, class_labels, shuffle=False, random_state=None):
        self.name = name
        # self._data = self.standardize_data(data)
        self._data = data
        self._target, self._classes, self._counts = self.standardize_targets(target)
        self._names = class_labels
        self._folds = (test_fold, nested_test_folds, len(np.unique(test_fold)), len(np.unique(nested_test_folds[0])))
        if shuffle:
            self.shuffle(random_state=random_state)

    def shuffle(self, random_state=None):
        self._data, self._target = skl_shuffle(self._data, self._target,
                                               random_state=random_state)

    def standardize_data(self, data):
        new_data = data.astype(float)
        data_mean = new_data.mean(axis=0)
        data_std = new_data.std(axis=0)
        data_std[data_std == 0] = 1
        return (new_data-data_mean)/data_std

    def standardize_targets(self, target):
        target = np.squeeze(target)
        classes, counts = np.unique(target, return_counts=True)
        return target, classes, counts

    def separate_sets(self, x, y, test_fold_id, test_folds):
        x_test = x[test_folds == test_fold_id, :]
        y_test = y[test_folds == test_fold_id]

        x_train = x[test_folds != test_fold_id, :]
        y_train = y[test_folds != test_fold_id]
        return [x_train, y_train, x_test, y_test]

    def reduce_number_instances(self, proportion=0.1):
        skf = StratifiedKFold(n_splits=int(1.0/proportion))
        test_folds = skf.test_folds
        train_idx, test_idx = next(iter(skf.split(X=self._data,
                                                  y=self._target)))
        self._data, self._target = self._data[test_idx], self._target[test_idx]

    @property
    def target(self):
        return self._target
    
    @property
    def folds(self):
        return self._folds

    #@target.setter
    #def target(self, new_value):
    #    self._target = new_value

    @property
    def data(self):
        return self._data

    @property
    def names(self):
        return self._names

    @property
    def classes(self):
        return self._classes

    @property
    def counts(self):
        return self._counts

    def print_summary(self):
        print(self)

    @property
    def n_classes(self):
        return len(self._classes)

    @property
    def n_features(self):
        return self._data.shape[1]

    @property
    def n_samples(self):
        return self._data.shape[0]

    def __str__(self):
        return("Name = {}\n"
               "Data shape = {}\n"
               "Target shape = {}\n"
               "Target classes = {}\n"
               "Target labels = {}\n"
               "Target counts = {}").format(self.name, self.data.shape,
                                            self.target.shape, self.classes,
                                            self.names, self.counts)


class Data:

    def __init__(self, random_state):
        self.datasets = {}

        # initialize wbc dataset
        WBC_DATA = "/home/maximl/Data/Experiment_data/newcastle/wbc/ideas_234_fluo.fcs"
        WBC_FOLDS = "/home/maximl/Data/Experiment_data/newcastle/wbc/samplesplit_234_nested_3fold.pkl"

        with open(WBC_FOLDS, "rb") as pkl:
            test_fold, nested_test_folds = pickle.load(pkl)

        data, y, sets, meta = loaders.load_features_from_ideas_fcs(WBC_DATA, keep_regex="(?i).*(BF|SSC|M01|M06|M09).*")
        data = cleaning.remove_unwanted(data)
        _, categorical, _ = organizing.group_features_per_category(data)
        data = data.drop(columns=categorical.columns)

        self.datasets['wbc'] = Dataset(
            name="wbc",
            data=data.values,
            target=y.values,
            test_fold=test_fold,
            nested_test_folds=nested_test_folds,
            class_labels=meta["class_labels"].split("|")
        )