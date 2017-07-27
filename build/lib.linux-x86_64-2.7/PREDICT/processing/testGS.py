from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.base import MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import indexable, check_is_fitted
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._split import check_cv
from sklearn.utils.fixes import rankdata
from sklearn.model_selection._validation import _fit_and_score
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals import six
from sklearn.utils.fixes import MaskedArray

from sklearn.model_selection._search import _CVScoreTuple, ParameterSampler

from abc import ABCMeta, abstractmethod
from collections import Sized, defaultdict
import numpy as np
from functools import partial
import warnings

import os
import random
import string
import fastr
import pandas as pd
import json


from sklearn.svm import SVC
import scipy
import glob


X = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
y = [0, 0, 0, 1, 1, 1]
groups = None
parameter_iterable = {'kernel': ['linear'], 'C': [0.5e8, 1e8], 'coef0': [0.5, 1],
                      'FeatSel_Variance': ['True']}
n_iter = 4
parameter_iterable = ParameterSampler(parameter_iterable, 4)


def fit(X, y, groups, parameter_iterable):
    """Actual fitting,  performing the search over parameters."""
    estimator = SVC(class_weight='balanced', probability=True)
    cv = 2
    scoring = 'f1_weighted'
    verbose = True
    fit_params = None
    return_train_score = True
    error_score = 'raise'

    estimator = estimator
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer_ = check_scoring(estimator, scoring=scoring)

    X, y, groups = indexable(X, y, groups)
    n_splits = cv.get_n_splits(X, y, groups)
    if verbose > 0 and isinstance(parameter_iterable, Sized):
        n_candidates = len(parameter_iterable)
        print("Fitting {0} folds for each of {1} candidates, totalling"
              " {2} fits".format(n_splits, n_candidates,
                                 n_candidates * n_splits))

    cv_iter = list(cv.split(X, y, groups))

    # Original: joblib
    # out = Parallel(
    #     n_jobs=n_jobs, verbose=verbose
    # )(delayed(_fit_and_score)(clone(base_estimator), X, y, scorer_,
    #                           train, test, verbose, parameters,
    #                           fit_params=fit_params,
    #                           return_train_score=return_train_score,
    #                           return_n_test_samples=True,
    #                           return_times=True, return_parameters=True,
    #                           error_score=error_score)
    #   for parameters in parameter_iterable
    #   for train, test in cv_iter)

    name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    tempfolder = os.path.join(fastr.config.mounts['tmp'], 'GS', name)
    if not os.path.exists(tempfolder):
        os.makedirs(tempfolder)

    # Create the parameter files
    parameter_files = dict()
    print parameter_iterable
    for num, parameters in enumerate(parameter_iterable):
        print parameters
        parameters["Number"] = str(num)

        # Convert parameter set to json
        fname = ('settings_{}.json').format(str(num))
        sourcename = os.path.join(tempfolder, 'parameters', fname)
        if not os.path.exists(os.path.dirname(sourcename)):
            os.makedirs(os.path.dirname(sourcename))

        with open(sourcename, 'w') as fp:
            json.dump(parameters, fp, indent=4)

        parameter_files[str(num)] = ('vfs://tmp/{}/{}/{}/{}').format('GS',
                                                                     name,
                                                                     'parameters',
                                                                     fname)

    # Create test-train splits
    traintest_files = dict()
    # TODO: ugly nummering solution
    num = 0
    for train, test in cv_iter:
        source_labels = ['train', 'test']

        source_data = pd.Series([train, test],
                                index=source_labels,
                                name='Train-test data')

        fname = ('traintest_{}.hdf5').format(str(num))
        sourcename = os.path.join(tempfolder, 'traintest', fname)
        if not os.path.exists(os.path.dirname(sourcename)):
            os.makedirs(os.path.dirname(sourcename))

        traintest_files[str(num)] = ('vfs://tmp/{}/{}/{}/{}').format('GS',
                                                                     name,
                                                                     'traintest',
                                                                     fname)

        sourcelabel = ("Source Data Iteration {}").format(str(num))
        source_data.to_hdf(sourcename, sourcelabel)

        num += 1

    # Create the files containing the estimator and settings
    estimator_labels = ['base_estimator', 'X', 'y', 'scorer',
                        'verbose', 'fit_params', 'return_train_score',
                        'return_n_test_samples',
                        'return_times', 'return_parameters',
                        'error_score']

    estimator_data = pd.Series([estimator, X, y, scorer_,
                                verbose,
                                fit_params, return_train_score,
                                True, True, True,
                                error_score],
                               index=estimator_labels,
                               name='estimator Data')
    fname = 'estimatordata.hdf5'
    estimatorname = os.path.join(tempfolder, fname)
    estimator_data.to_hdf(estimatorname, 'Estimator Data')

    estimatordata = ("vfs://tmp/{}/{}/{}").format('GS', name, fname)

    # Create the fastr network
    network = fastr.Network('GridSearch_' + name)
    estimator_data = network.create_source('HDF5', id_='estimator_source')
    traintest_data = network.create_source('HDF5', id_='traintest')
    parameter_data = network.create_source('JsonFile', id_='parameters')
    sink_output = network.create_sink('HDF5', id_='output')

    fitandscore = network.create_node('fitandscore', memory='2G', id_='fitandscore')
    fitandscore.inputs['estimatordata'].input_group = 'estimator'
    fitandscore.inputs['traintest'].input_group = 'traintest'
    fitandscore.inputs['parameters'].input_group = 'parameters'

    fitandscore.inputs['estimatordata'] = estimator_data.output
    fitandscore.inputs['traintest'] = traintest_data.output
    fitandscore.inputs['parameters'] = parameter_data.output
    sink_output.input = fitandscore.outputs['fittedestimator']

    source_data = {'estimator_source': estimatordata,
                   'traintest': traintest_files,
                   'parameters': parameter_files}
    sink_data = {'output': ("vfs://tmp/{}/{}/output_{{sample_id}}_{{cardinality}}{{ext}}").format('GS', name)}

    network.draw_network(network.id, draw_dimension=True)
    print source_data
    network.execute(source_data, sink_data, tmpdir=os.path.join(tempfolder, 'fastr'))

    # Read in the output data once finished
    # TODO: expanding fastr url is probably a nicer way
    sink_files = glob.glob(os.path.join(fastr.config.mounts['tmp'],'GS', name) + '/output*.hdf5')
    save_data = list()
    features_labels = list()
    for output in sink_files:
        data = pd.read_hdf(output)

        temp_save_data = data['RET']

        save_data.append(temp_save_data)
        features_labels.append(data['feature_labels'])

    # if one choose to see train score, "out" will contain train score info
    if return_train_score:
        (train_scores, test_scores, test_sample_counts,
         fit_time, score_time, parameters) = zip(*save_data)
    else:
        (test_scores, test_sample_counts,
         fit_time, score_time, parameters) = zip(*save_data)

    candidate_params = parameters[::n_splits]
    n_candidates = len(candidate_params)

    results = dict()

    def _store(key_name, array, weights=None, splits=False, rank=False):
        """A small helper to store the scores/times to the cv_results_"""
        array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                          n_splits)
        if splits:
            for split_i in range(n_splits):
                results["split%d_%s"
                        % (split_i, key_name)] = array[:, split_i]

        array_means = np.average(array, axis=1, weights=weights)
        results['mean_%s' % key_name] = array_means
        # Weighted std is not directly available in numpy
        array_stds = np.sqrt(np.average((array -
                                         array_means[:, np.newaxis]) ** 2,
                                        axis=1, weights=weights))
        results['std_%s' % key_name] = array_stds

        if rank:
            results["rank_%s" % key_name] = np.asarray(
                rankdata(-array_means, method='min'), dtype=np.int32)

    # Computed the (weighted) mean and std for test scores alone
    # NOTE test_sample counts (weights) remain the same for all candidates
    test_sample_counts = np.array(test_sample_counts[:n_splits],
                                  dtype=np.int)

    _store('test_score', test_scores, splits=True, rank=True,
           weights=test_sample_counts if iid else None)
    if return_train_score:
        _store('train_score', train_scores, splits=True)
    _store('fit_time', fit_time)
    _store('score_time', score_time)

    best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
    best_parameters = candidate_params[best_index]

    # Use one MaskedArray and mask all the places where the param is not
    # applicable for that candidate. Use defaultdict as each candidate may
    # not contain all the params
    param_results = defaultdict(partial(MaskedArray,
                                        np.empty(n_candidates,),
                                        mask=True,
                                        dtype=object))
    for cand_i, params in enumerate(candidate_params):
        for name, value in params.items():
            # An all masked empty array gets created for the key
            # `"param_%s" % name` at the first occurence of `name`.
            # Setting the value at an index also unmasks that index
            param_results["param_%s" % name][cand_i] = value

    results.update(param_results)

    # Store a list of param dicts at the key 'params'
    results['params'] = candidate_params

    cv_results_ = results
    best_index_ = best_index
    n_splits_ = n_splits

    if refit:
        # fit the best estimator using the entire dataset
        # clone first to work around broken estimators
        best_estimator = clone(base_estimator).set_params(
            **best_parameters)
        if y is not None:
            best_estimator.fit(X, y, **fit_params)
        else:
            best_estimator.fit(X, **fit_params)
        best_estimator_ = best_estimator
    return self

fit(X, y, groups, parameter_iterable)
