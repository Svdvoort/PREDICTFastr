# Copyright 2017-2018 Biomedical Imaging Group Rotterdam, Departments of
# Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import pandas as pd
import PREDICT.IOparser.config_io_classifier as config_io
import PREDICT.genetics.genetic_processing as gp
import json
import os
import sklearn

from PREDICT.classification import crossval as cv
from PREDICT.classification import construct_classifier as cc
from PREDICT.plotting.plot_SVM import plot_single_SVM
from PREDICT.plotting.plot_SVR import plot_single_SVR


def trainclassifier_old(feat_m1, patientinfo, config,
                    output_svm, output_json,
                    feat_m2=None, feat_m3=None,
                    fixedsplits=None, verbose=True):
    # Load variables from the config file
    config = config_io.load_config(config)

    if type(patientinfo) is list:
        patientinfo = ''.join(patientinfo)

    if type(config) is list:
        config = ''.join(config)

    label_type = config['Genetics']['mutation_type']

    # Read the features and classification data
    label_data, image_features =\
        readdata(feat_m1, feat_m2, feat_m3, patientinfo,
                 label_type)

    # Create tempdir name from patientinfo file name
    basename = os.path.basename(patientinfo)
    filename, _ = os.path.splitext(basename)
    path = patientinfo
    for i in range(4):
        # Use temp dir: result -> sample# -> parameters - > temppath
        path = os.path.dirname(path)

    _, path = os.path.split(path)
    path = os.path.join(path, 'trainclassifier', filename)

    # Construct the required classifier
    classifier, param_grid =\
        cc.construct_classifier(config,
                                image_features[0][0])

    # Append the feature groups to the parameter grid
    if config['General']['FeatureCalculator'] == 'CalcFeatures':
        param_grid['SelectGroups'] = 'True'
        for group in config['SelectFeatGroup'].keys():
            param_grid[group] = config['SelectFeatGroup'][group]

    if config['FeatureScaling']['scale_features']:
        if type(config['FeatureScaling']['scaling_method']) is not list:
            param_grid['FeatureScaling'] = [config['FeatureScaling']['scaling_method']]
        else:
            param_grid['FeatureScaling'] = config['FeatureScaling']['scaling_method']

    param_grid['Featsel_Variance'] = config['Featsel']['Variance']

    # For N_iter, perform k-fold crossvalidation
    trained_classifier = cv.crossval(config, label_data,
                                     image_features,
                                     classifier, param_grid,
                                     use_fastr=config['Classification']['fastr'],
                                     fixedsplits=fixedsplits)

    if type(output_svm) is list:
        output_svm = ''.join(output_svm)

    if not os.path.exists(os.path.dirname(output_svm)):
        os.makedirs(os.path.dirname(output_svm))

    trained_classifier.to_hdf(output_svm, 'SVMdata')

    # Calculate statistics of performance
    if type(classifier) == sklearn.svm.SVR:
        statistics = plot_single_SVR(trained_classifier, label_data,
                                     label_type)
    else:
        statistics = plot_single_SVM(trained_classifier, label_data,
                                     label_type)

    # Save output
    savedict = dict()
    savedict["Statistics"] = statistics

    if type(output_json) is list:
        output_json = ''.join(output_json)

    if not os.path.exists(os.path.dirname(output_json)):
        os.makedirs(os.path.dirname(output_json))

    with open(output_json, 'w') as fp:
        json.dump(savedict, fp, indent=4)

    print("Saved data!")

def readdata(feat_m1, feat_m2, feat_m3, patientinfo, mutation_type):
    # Read and stack the features
    image_features = list()
    for i_feat in range(len(feat_m1)):
        if feat_m1 is not None:
            feat_temp = pd.read_hdf(feat_m1[i_feat])
            feature_values_temp = feat_temp.feature_values
            feature_labels_temp = feat_temp.feature_labels

            # Combine modalities
            if feat_m2 is not None:
                # First rename the M1 labels
                feature_labels_temp = [f + '_M1' for f in feature_labels_temp]

                # Append the M2 values and labels
                feat_temp = pd.read_hdf(feat_m2[i_feat])
                feature_values_temp += feat_temp.feature_values
                feature_labels_temp += [f + '_M2' for f in feat_temp.feature_labels]

            if feat_m3 is not None:
                # Append the M3 values and labels
                feat_temp = pd.read_hdf(feat_m3[i_feat])
                feature_values_temp += feat_temp.feature_values
                feature_labels_temp += [f + '_M3' for f in feat_temp.feature_labels]

            image_features.append((feature_values_temp, feature_labels_temp))

    # Get the mutation labels and patient IDs
    mutation_data, image_features =\
        gp.findmutationdata(patientinfo,
                            mutation_type,
                            feat_m1,
                            image_features)

    print("Mutation Labels:")
    print(mutation_data['mutation_label'])
    print('Total of ' + str(mutation_data['patient_IDs'].shape[0]) +
          ' patients')
    pos = np.sum(mutation_data['mutation_label'])
    neg = mutation_data['patient_IDs'].shape[0] - pos
    print(('{} positives, {} negatives').format(pos, neg))

    return mutation_data, image_features
