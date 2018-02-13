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
import os
from scipy.stats import ttest_ind, ranksums, mannwhitneyu
import csv


def ttest(features, patientinfo, config, output, verbose=True):
    # Load variables from the config file
    config = config_io.load_config(config)

    if type(patientinfo) is list:
        patientinfo = ''.join(patientinfo)

    if type(config) is list:
        config = ''.join(config)

    if type(output) is list:
        output = ''.join(output)

    # Create output folder if required
    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))

    label_type = config['Genetics']['mutation_type']

    # Read the features and classification data
    print("Reading features and label data.")
    label_data, image_features =\
        readsingledata(features, patientinfo,
                       label_type)

    # Extract feature labels and put values in an array
    feature_labels = image_features[0][1]
    feature_values = np.zeros([len(image_features), len(feature_labels)])
    for num, x in enumerate(image_features):
        feature_values[num, :] = x[0]

    # Open the output file
    myfile = open(output, 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

    # Do a t-test on all feature values
    label_value = label_data['mutation_label']
    label_name = label_data['mutation_name']

    header = list()
    subheader = list()
    for i_name in label_name:
        header.append(str(i_name[0]))
        header.append('')
        header.append('')
        header.append('')
        # header.append('')
        header.append('')

        subheader.append('Label')
        subheader.append('Ttest')
        subheader.append('Welch')
        subheader.append('Wilcoxon')
        # subheader.append('Mann-Whitney')
        subheader.append('')

    wr.writerow(header)
    wr.writerow(subheader)

    savedict = dict()
    for i_class, i_name in zip(label_value, label_name):
        savedict[i_name[0]] = dict()
        pvalues = list()
        pvalueswelch = list()
        pvalueswil = list()
        # pvaluesmw = list()

        for num, fl in enumerate(feature_labels):
            fv = feature_values[:, num]
            classlabels = i_class.ravel()

            class1 = [i for j, i in enumerate(fv) if classlabels[j] == 1]
            class2 = [i for j, i in enumerate(fv) if classlabels[j] == 0]

            pvalues.append(ttest_ind(class1, class2)[1])
            pvalueswelch.append(ttest_ind(class1, class2, equal_var=False)[1])
            pvalueswil.append(ranksums(class1, class2)[1])
            # pvaluesmw.append(mannwhitneyu(class1, class2)[1])

        # Sort based on p-values:
        pvalues = np.asarray(pvalues)
        indices = np.argsort(pvalues)
        pvalues = pvalues[indices].tolist()
        feature_labels_o = np.asarray(feature_labels)[indices].tolist()
        pvalueswelch = np.asarray(pvalueswelch)[indices].tolist()
        pvalueswil = np.asarray(pvalueswil)[indices].tolist()
        # pvaluesmw = np.asarray(pvaluesmw)[indices].tolist()

        savedict[i_name[0]]['ttest'] = pvalues
        savedict[i_name[0]]['welch'] = pvalueswelch
        savedict[i_name[0]]['wil'] = pvalueswil
        # savedict[i_name[0]]['mw'] = pvaluesmw
        savedict[i_name[0]]['labels'] = feature_labels_o

    for num in range(0, len(savedict[i_name[0]]['ttest'])):
        writelist = list()
        for i_name in savedict.keys():
            labeldict = savedict[i_name]
            writelist.append(labeldict['labels'][num])
            writelist.append(labeldict['ttest'][num])
            writelist.append(labeldict['welch'][num])
            writelist.append(labeldict['wil'][num])
            # writelist.append(labeldict['mw'][num])
            writelist.append('')

        wr.writerow(writelist)

    print("Saved data!")


def readsingledata(featurefiles, patientinfo=None, mutation_type=None):
    # Read and stack the features
    image_features = list()
    for i_feat in range(len(featurefiles)):
        feat_temp = pd.read_hdf(featurefiles[i_feat])
        feature_values_temp = feat_temp.feature_values
        feature_labels_temp = feat_temp.feature_labels
        image_features.append((feature_values_temp, feature_labels_temp))

    # Get the mutation labels and patient IDs
    if patientinfo is not None:
        mutation_data, image_features =\
            gp.findmutationdata(patientinfo,
                                mutation_type,
                                featurefiles,
                                image_features)

        print("Mutation Labels:")
        print(mutation_data['mutation_label'])
        print('Total of ' + str(mutation_data['patient_IDs'].shape[0]) +
              ' patients')
        pos = np.sum(mutation_data['mutation_label'])
        neg = mutation_data['patient_IDs'].shape[0] - pos
        print(('{} positives, {} negatives').format(pos, neg))
    else:
        # Use filenames as patient ID s
        patient_IDs = list()
        for i in featurefiles:
            patient_IDs.append(os.path.basename(i))
        mutation_data = dict()
        mutation_data['patient_IDs'] = patient_IDs

    return mutation_data, image_features
