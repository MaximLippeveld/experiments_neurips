# Usage:
# Parallelized in multiple threads: #   python -m scoop -n 4 main.py # where -n is the number of workers ( # threads)
# Not parallelized (easier to debug):
#   python main.py
from __future__ import division
import sys
import argparse
import os
import pandas
import numpy as np

# Classifiers
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import calib.models.adaboost as our
from calib.models.classifiers import MockClassifier
import sklearn.ensemble as their
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.dummy import DummyClassifier

# ifc libs
from ifclibs import training, loaders, cleaning

# Parallelization
import itertools
#import scoop
#from scoop import futures, shared
from multiprocessing import cpu_count, Pool

# Our classes and modules
from calib.utils.calibration import cv_calibration
from calib.utils.dataframe import MyDataFrame
from calib.utils.functions import get_sets
from calib.utils.functions import p_value
from calib.utils.functions import serializable_or_string
from calib.models.calibration import MAP_CALIBRATORS

from calib.utils.summaries import create_summary_path
from calib.utils.summaries import generate_summaries
from calib.utils.summaries import generate_summary_hist
from calib.utils.plots import export_boxplot
from calib.utils.plots import plot_reliability_diagram_per_class
from calib.utils.plots import plot_multiclass_reliability_diagram
from calib.utils.plots import save_fig_close

# Our datasets module
from data_wrappers.datasets import Data

import logging

classifiers = {
      'mock': DummyClassifier(strategy="prior"),
      'nbayes': GaussianNB(),
      'logistic': LogisticRegression(random_state=42),
      #'adao': our.AdaBoostClassifier(n_estimators=200),
      'adas': their.AdaBoostClassifier(n_estimators=200, random_state=42),
      'forest': RandomForestClassifier(n_estimators=200, random_state=42),
      'mlp': MLPClassifier(random_state=42),
      'svm': SVC(probability=True, random_state=42),
      'knn': KNeighborsClassifier(3),
      'svc-linear': SVC(kernel="linear", C=0.025, probability=True, random_state=42),
      'svc-rbf': SVC(gamma=2, C=1, probability=True, random_state=42),
      'gp': GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
      'tree': DecisionTreeClassifier(max_depth=5, random_state=42),
      'qda': QuadraticDiscriminantAnalysis(reg_param=0.1),
      'lda': LinearDiscriminantAnalysis()
}

score_types = {
      'mock': 'predict_proba',
      'nbayes': 'predict_proba',
      'logistic': 'predict_proba',
      #'adao': 'predict_proba',
      'adas': 'predict_proba',
      'forest': 'predict_proba',
      'mlp': 'predict_proba',
      'svm': 'sigmoid',
      'knn': 'predict_proba',
      'svc-linear': 'predict_proba',
      'svc-rbf': 'predict_proba',
      'gp': 'predict_proba',
      'tree': 'predict_proba',
      'qda': 'predict_proba',
      'lda': 'predict_proba'
}

columns = ['dataset', 'n_classes', 'n_features', 'n_samples', 'method', 'mc',
           'test_fold', 'train_acc', 'train_loss', 'train_brier',
           'train_guo-ece', 'train_cla-ece', 'train_full-ece',
           'train_mce',
           'acc', 'loss', 'brier', 'guo-ece', 'cla-ece', 'full-ece',
           'p-guo-ece', 'p-cla-ece', 'p-full-ece', 'mce',
           'confusion_matrix', 'c_probas', 'y_test', 'exec_time',
           'calibrators']

save_columns = [c for c in columns if c not in ['c_probas', 'y_test']]


def comma_separated_strings(s):
    try:
        return s.split(',')
    except ValueError:
        msg = "Not a valid comma separated list: {}".format(s)
        raise argparse.ArgumentTypeError(msg)


def parse_arguments():
    parser = argparse.ArgumentParser(description='''Runs all the experiments
                                     with the given arguments''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--classifiers', dest='classifier_names',
                        type=comma_separated_strings,
                        default=['logistic', 'tree'],
                        help='''Classifiers to use for evaluation in a comma
                        separated list of strings. From the following
                        options: ''' + ', '.join(classifiers.keys()))
    parser.add_argument('-s', '--seed', dest='seed_num', type=int,
                        default=42,
                        help='Seed for the random number generator')
    parser.add_argument('-i', '--iterations', dest='mc_iterations', type=int,
                        default=2,
                        help='Number of Markov Chain iterations')
    parser.add_argument('-o', '--output-path', dest='results_path', type=str,
                        default='results_test',
                        help='''Path to store all the results''')
    parser.add_argument('-v', '--verbose', dest='verbose',
                        type=int, default=logging.INFO,
                        help='''Show additional messages, from 10 (debug) to
                        50 (fatal)''')
    parser.add_argument('-m', '--methods', dest='methods',
                        type=comma_separated_strings,
                        default=['uncalibrated', 'isotonic',
                                 'dirichlet_full_l2'],
                        help=('Comma separated calibration methods from ' +
                              'the following options: ' +
                              ', '.join(MAP_CALIBRATORS.keys())))
    parser.add_argument('-w', '--workers', dest='n_workers', type=int,
                        default=-1,
                        help='''Number of jobs to run concurrently. -1 to use all
                                available CPUs''')
    return parser.parse_args()


def compute_all(args):
    ''' Train a classifier with the specified dataset and calibrate

    Parameters
    ----------
    args is a tuple with all the following:

    name : string
        Name of the dataset to use
    folds : string
        Pickle file containing outer and inner folds.
    mc : int
        Monte Carlo repetition index, in order to set different seeds to
        different repetitions, but same seed in calibrators in the same Monte
        Carlo repetition.
    classifier_name : string
        Name of the classifier to be trained and tested
    methods : string, or list of strings
        List of calibrators to be trained and tested
    verbose : int
        Integer indicating the verbosity level

    Returns
    -------
    df : pands.DataFrame
        DataFrame with the overall results of every calibration method
        d_name : string
            Name of the dataset
        method : string
            Calibrator method
        mc : int
            Monte Carlo repetition index
        acc : float
            Mean accuracy for the inner folds
        loss : float
            Mean Log-loss for the inner folds
        brier : float
            Mean Brier score for the inner folds
        mean_probas : array of floats (n_samples_test, n_classes)
            Mean probability predictions for the inner folds and the test set
        exec_time : float
            Mean calibration time for the inner folds
    '''
    logger = logging.getLogger(__name__)

    (dataset, test_fold, nested_test_folds, mc, classifier_name, methods, verbose) = args
    if isinstance(methods, str):
        methods = (methods,)
    classifier = classifiers[classifier_name]
    score_type = score_types[classifier_name]

    skf = training.RandomOversamplingPredefinedSplit(folds=test_fold, indices=True)
    df = []
    class_counts = np.bincount(dataset.target)
    t = dataset.target
    fold_id = 0
    n_folds = skf.get_n_splits()
    for i, ((train_idx, test_idx), nested_test_fold) in enumerate(zip(skf.split(X=dataset.data,
                                                        y=dataset.target), nested_test_folds)):
        
        logger.info(f'{classifier_name}, {dataset}, {mc}, {str(methods)}: outer fold {i+1} of {n_folds}')

        x_train, y_train = dataset.data[train_idx], dataset.target[train_idx]
        x_test, y_test = dataset.data[test_idx], dataset.target[test_idx]

        cv = training.RandomOversamplingPredefinedSplit(folds=nested_test_fold, indices=True)

        results = cv_calibration(classifier, methods, x_train, y_train, x_test,
                                 y_test, cv=cv, score_type=score_type,
                                 verbose=verbose, seed=mc)
        (train_acc, train_loss, train_brier, train_guo_ece, train_cla_ece,
         train_full_ece, train_mce, accs, losses, briers,
         guo_eces, cla_eces, full_eces, p_guo_eces, p_cla_eces, p_full_eces,
         mces, cms, mean_probas, cl, exec_time) = results

        for method in methods:
            df.append([dataset.name, dataset.n_classes,
                                  dataset.n_features, dataset.n_samples,
                                  method, mc, fold_id, train_acc[method],
                                  train_loss[method], train_brier[method],
                                  train_guo_ece[method], train_cla_ece[method],
                                  train_full_ece[method], train_mce[method],
                                  accs[method], losses[method], briers[method],
                                  guo_eces[method], cla_eces[method],
                                  full_eces[method], p_guo_eces[method],
                                  p_cla_eces[method], p_full_eces[method],
                                  mces[method], cms[method],
                                  mean_probas[method], y_test,
                                  exec_time[method],
                                  [{key: serializable_or_string(value) for key, value in
                                      c.calibrator.__dict__.items()} for c in cl[method]]
                                  ])

        fold_id += 1
    return pandas.DataFrame(df, columns=columns)


# FIXME seed_num is not being used at the moment
def main(seed_num, mc_iterations, classifier_names, results_path,
		 verbose, methods, n_workers, fig_titles=False):
    if not fig_titles:
        title = None

    # setup logging
    logging.basicConfig(format="%(levelname)s:%(asctime)s|%(name)s|%(process)d - %(message)s", level=verbose)
    logging.captureWarnings(True)
    warn_logger = logging.getLogger('py.warnings')
    warn_logger.propagate = False
    warn_handler = logging.FileHandler("warnings.log")
    warn_logger.addHandler(warn_handler)

    logger = logging.getLogger(__name__)

    logger.debug(locals())

    columns_hist = ['classifier', 'dataset', 'calibration'] + \
                   ['{}-{}'.format(i/10, (i+1)/10) for i in range(0,10)]
    
    data = Data(random_state=seed_num)

    results_path_root = results_path
    for classifier_name in classifier_names:
        results_path = os.path.join(results_path_root, classifier_name)

        for name, dataset in data.datasets.items():
            logger.info(dataset)

            test_fold, nested_test_folds, n_folds, n_inner_folds = dataset.folds

            df = MyDataFrame(columns=columns)
            # Assert that every class has enough samples to perform the two
            # cross-validataion steps (classifier + calibrator)
            smaller_count = min(dataset.counts)
            if (smaller_count < n_folds) or \
               ((smaller_count*(n_folds-1)/n_folds) < n_inner_folds):
                raise ValueError(("At least one of the classes does not have enough "
                             "samples for outer {} folds and inner {} folds"
                            ).format(n_folds, n_inner_folds))

            mcs = np.arange(mc_iterations)
            # All the arguments as a list of lists
            args = [[dataset], [test_fold], [nested_test_folds], mcs, [classifier_name],
                    methods, [verbose]]
            args = list(itertools.product(*args))

            logger.info('There are ' + str(len(args)) + ' sets of arguments that need to be run')
            logger.debug('The following is a list with all the arguments')
            logger.debug(args)

            if n_workers == -1:
                n_workers = cpu_count()

            if n_workers == 1:
                dfs = map(compute_all, args)
            else:
                if n_workers > len(args):
                    n_workers = len(args)

                with Pool(n_workers) as pool:
                    logger.info('{} jobs will be deployed in {} workers'.format(len(args), n_workers))
                    dfs = pool.map(compute_all, args)

            logger.info("All results are collected.")

            df = df.concat(dfs)

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            # Export score distributions for dataset + classifier + calibrator
            logger.info("Exporting score distributions")
            def MakeList(x):
                T = tuple(x)
                if len(T) > 1:
                    return T
                else:
                    return T[0]
            #df_scores = df.drop_duplicates(subset=['dataset', 'method'])
            g = df.groupby(['dataset', 'method'])
            df_scores = g.agg({'y_test': MakeList,
                               'c_probas': MakeList,
                               'n_classes': 'max',
                               'method': 'first',
                               'loss': 'mean',
                               'brier': 'mean',
                               'acc': 'mean',
                               'guo-ece': 'mean',
                               'cla-ece': 'mean',
                               'full-ece': 'mean',
                               'p-guo-ece': MakeList,
                               'p-cla-ece': MakeList,
                               'p-full-ece': MakeList,
                               'mce': 'mean'})
            for index, row in df_scores.iterrows():
                filename = os.path.join(results_path, '_'.join([classifier_name,
                                                                name,
                                                                row['method'],
                                                                'positive_scores']))
                y_test = np.hstack(row['y_test'])
                if fig_titles:
                    title = (("{}, test samples = {}, {}\n"
                          "acc = {:.2f}, log-loss = {:.2e},\n"
                          "brier = {:.2e}, full-ece = {:.2e}, mce = {:.2e}")
                           .format(name, len(y_test),
                                   row['method'], row['acc'],
                                   row['loss'], row['brier'], row['full-ece'],
                                   row['mce']))
                try:
                    export_boxplot(method = row['method'],
                                   scores = np.vstack(row['c_probas']),
                                   y_test = y_test,
                                   n_classes = row['n_classes'],
                                   name_classes = dataset.names,
                                   title = title,
                                   per_class = False,
                                   figsize=(int(row['n_classes']/2), 2),
                                   filename=filename, file_ext='.svg')

                    export_boxplot(method = row['method'],
                                   scores = np.vstack(row['c_probas']),
                                   y_test = y_test,
                                   n_classes = row['n_classes'],
                                   name_classes = dataset.names,
                                   title = title,
                                   per_class = True,
                                   figsize=(int(row['n_classes']/2), 1+row['n_classes']),
                                   filename=filename + '_per_class', file_ext='.svg')
                except Error as e:
                    print(e)


                #scores = [row['c_probas'][row['y_test'] == i].flatten() for i in
                #                   range(row['n_classes'])]

            # Export reliability diagrams per dataset + classifier + calibrator
            logger.info("Exporting reliability diagrams")
            g = df.groupby(['dataset', 'method'])
            df_scores = g.agg({'y_test': MakeList,
                               'c_probas': MakeList,
                               'n_classes': 'max'})
            for index, row in df_scores.iterrows():
                y_test = label_binarize(np.hstack(row['y_test']),
                                        classes=range(row['n_classes']))
                p_pred = np.vstack(row['c_probas'])
                try:
                    filename = os.path.join(results_path, '_'.join([classifier_name,
                                                                name,
                                                                index[1],
                                                                'rel_diagr_perclass']))
                    fig = plot_reliability_diagram_per_class(y_true=y_test,
                                                             p_pred=p_pred)
                    save_fig_close(fig, filename + '.svg')

                    filename = os.path.join(results_path, '_'.join([classifier_name,
                                                                name,
                                                                index[1],
                                                                'rel_diagr']))
                    fig = plot_multiclass_reliability_diagram(y_true=y_test,
                                                              p_pred=p_pred)
                    save_fig_close(fig, filename + '.svg')

                    y_labels = np.hstack(row['y_test'])
                    y_pred = p_pred.argmax(axis=1)
                    y_conf = (y_labels == y_pred).astype(int)
                    p_conf_pred = p_pred.max(axis=1)
                    filename = os.path.join(results_path, '_'.join([classifier_name,
                                                                name,
                                                                index[1],
                                                                'conf_rel_diagr']))
                    fig = plot_multiclass_reliability_diagram(
                        y_true=y_conf, p_pred=p_conf_pred,
                        labels=['Obs.  accuracy', 'Gap pred. mean'])
                    save_fig_close(fig, filename + '.svg')

                except:
                    print("Unexpected error:" + sys.exc_info()[0])

            for method in methods:
                df[df['method'] == method][save_columns].to_csv(
                    os.path.join(results_path, '_'.join([classifier_name, name,
                                                         method,
                                                         'raw_results.csv'])))

            logger.info('Saving histogram of all the scores')
            for method in methods:
                hist = np.histogram(np.concatenate(
                            df[df.dataset == name][df.method ==
                                                   method]['c_probas'].values),
                            range=(0.0, 1.0))
                df_hist = MyDataFrame(data=[[classifier_name, name, method] +
                                           hist[0].tolist()],
                                      columns=columns_hist)
                df_hist.to_csv(os.path.join(results_path, '_'.join(
                    [classifier_name, name, method, 'score_histogram.csv'])))


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
