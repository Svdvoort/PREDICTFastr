# Copyright 2011-2017 Biomedical Imaging Group Rotterdam, Departments of
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
from sklearn.feature_selection import VarianceThreshold

from PREDICT.classification import crossval as cv
from PREDICT.classification import construct_classifier as cc
from PREDICT.tools.plot_SVM import plot_single_SVM


def trainclassifier(feat_m1, feat_m2, feat_m3, patientinfo, config,
                    parameter_file, output_svm, output_json, verbose=True):
    # Load variables from the config file
    config = config_io.load_config(config)
    if type(parameter_file) is list:
        parameter_file = ''.join(parameter_file)

    if type(patientinfo) is list:
        patientinfo = ''.join(patientinfo)

    if type(config) is list:
        config = ''.join(config)

    with open(parameter_file) as data_file:
        parameters = json.load(data_file)

    label_type = config['Genetics']['mutation_type']

    # Read the features and classification data
    image_features_select, labels, label_data =\
        readdata(feat_m1, feat_m2, feat_m3, patientinfo,
                 label_type, parameters)

    # Delete features which are are the same in more than 99% of patients
    # TODO: Separate this into a different tool
    sel = VarianceThreshold(threshold=0.99*(1 - 0.99))
    sel = sel.fit(image_features_select)
    image_features_select = sel.transform(image_features_select)
    labels = sel.transform(labels).tolist()[0]

    # If we have too few features left, don't proceed
    if len(image_features_select[1]) > 7:

        # Create tempdir name from parameter file name
        basename = os.path.basename(parameter_file)
        filename, _ = os.path.splitext(basename)
        path = parameter_file
        for i in range(4):
            # Use temp dir: result -> sample# -> parameters - > temppath
            path = os.path.dirname(path)

        _, path = os.path.split(path)
        path = os.path.join(path, 'trainclassifier', filename)

        # Construct the required classifier
        classifier, param_grid =\
            cc.construct_classifier(config,
                                    image_features_select[0])

        # For N_iter, perform k-fold crossvalidation
        if config['Classification']['fastr']:
            trained_classifier = cv.crossvalfastr(config, label_data,
                                                  image_features_select,
                                                  classifier, param_grid, path)
        else:
            trained_classifier = cv.crossval(config, label_data,
                                             image_features_select,
                                             classifier, param_grid, path)
        # Add labels to dataframe
        # TODO: Works only if single mutation is present
        labels_pd =\
            pd.Series([labels],
                      index=[trained_classifier.keys()[0]],
                      name='feature_labels')
        classifier = classifier.append(labels_pd)

        # Calculate statistics of performance
        statistics = plot_single_SVM(classifier, label_data)

    else:
        statistics = "None"

        labels = ["Too Few Features."]
        feat = ["None"]

        panda_dict = dict(zip(labels, feat))

        classifier = pd.Series(panda_dict)

    # Save output
    savedict = dict()
    savedict["Parameters"] = parameters
    savedict["Statistics"] = statistics

    print("Saving data!")
    if type(output_svm) is list:
        output_svm = ''.join(output_svm)

    if type(output_json) is list:
        output_json = ''.join(output_json)

    # TODO: ouptu_svm/json are list objects!
    classifier.to_hdf(output_svm, 'SVMdata')
    with open(output_json, 'w') as fp:
        json.dump(savedict, fp, indent=4)


def readdata(feat_m1, feat_m2, feat_m3, patientinfo, mutation_type,
             parameters):
    # Read and stack the features
    image_features_temp = list()
    for i_feat in range(len(feat_m1)):
        if feat_m1 is not None:
            feat_temp_m1 = pd.read_hdf(feat_m1[i_feat])
            feat_temp = feat_temp_m1.image_features

            # Combine modalities
            if feat_m2 is not None:
                feat_temp_m2 = pd.read_hdf(feat_m2[i_feat])
                feat_temp_m2 = feat_temp_m2.image_features
                for label in ['shape_features', 'orientation_features',
                              'histogram_features', 'texture_features',
                              'patient_features', 'semantic_features']:
                    feat_temp[label].append(feat_temp_m2[label])

            if feat_m3 is not None:
                feat_temp_m3 = pd.read_hdf(feat_m3[i_feat])
                feat_temp_m3 = feat_temp_m3.image_features
                # TODO: Hacked out the semantic and patient features
                for label in ['shape_features', 'orientation_features',
                              'histogram_features', 'texture_features',
                              'patient_features', 'semantic_features']:
                    feat_temp[label + '_m3'] = feat_temp_m3[label]

            image_features_temp.append(feat_temp)

    # Get the mutation labels and patient IDs
    mutation_data, image_features = gp.findmutationdata(patientinfo,
                                                        mutation_type,
                                                        feat_m1,
                                                        image_features_temp)

    print("Mutation Labels:")
    print(mutation_data['mutation_label'])
    print('Total of ' + str(mutation_data['patient_IDs'].shape[0]) +
          ' patients')
    pos = np.sum(mutation_data['mutation_label'])
    neg = mutation_data['patient_IDs'].shape[0] - pos
    print(('{} positives, {} negatives').format(pos, neg))

    # Select only features specificed by parameters per patient
    image_features_select = list()
    image_features_temp = dict()
    for num, feat in enumerate(image_features):
        for feattype in parameters.keys():
            if feattype in feat.keys():
                if parameters[feattype] == 'True':
                    success = False
                    for k in feat.keys():
                        if feattype in k:
                            image_features_temp[k] = feat[k]
                            success = True

                    if not success:
                        print(("No {} present!").format(feattype))
                        parameters[feattype] = 'False'
                elif parameters[feattype] == 'False':
                    # Do not use the features
                    pass
                else:
                    success = False
                    for k in feat[feattype].keys():
                        print parameters[feattype], k
                        if parameters[feattype] in k:
                            success = True
                            # Only use the type given
                            feat_dict = dict()
                            feat_dict[k] = feat[feattype][parameters[feattype]]
                            feat_series = pd.Series(feat_dict)
                            image_features_temp[k] = feat_series

                    if not success:
                        print(("No {} present!").format(feattype))
                        parameters[feattype] = 'False'
            else:
                if "Number" not in feattype:
                    print(("{} is not a valid feature type.").format(feattype))

        # Stack remainging feature dictionaries in array
        image_feature_array = list()

        for _, feattype in image_features_temp.iteritems():
            for _, features in feattype.iteritems():
                image_feature_array.extend(features.values)

        if num == 1:
            # Use first patient for label readout
            labelfeat = image_features_temp.copy()

        image_features_select.append(image_feature_array)

    # Save the feature labels
    labels_temp = list()

    for _, feattype in labelfeat.iteritems():
        for _, features in feattype.iteritems():
            keys = features.keys().tolist()
            for k in keys:
                num = 1
                label = k + '_' + str(num)
                while label in labels_temp:
                    num += 1
                    label = k + '_' + str(num)

                labels_temp.append(label)

    # Convert list of patient features to array
    image_features_select = np.asarray(image_features_select)

    # Remove NAN
    delrows = list()
    labels = list()
    for i in range(len(image_features_select[0])):
        if np.isnan(image_features_select[0][i]):
            delrows.append(i)
        else:
            labels.append(labels_temp[i])

    image_features_select = np.delete(image_features_select, delrows, 1)

    return image_features_select, labels, mutation_data
