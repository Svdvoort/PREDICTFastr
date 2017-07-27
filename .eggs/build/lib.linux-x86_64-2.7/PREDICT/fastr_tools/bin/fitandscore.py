#!/usr/bin/env python

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

from sklearn.model_selection._validation import _fit_and_score
import argparse
import json
import pandas as pd
# from PREDICT.featureselection.selfeat import selectgroups, selfeat_variance
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def selfeat_variance(image_features, labels, thresh=0.99):
    sel = VarianceThreshold(threshold=thresh*(1 - thresh))
    sel = sel.fit(image_features)
    image_features = sel.transform(image_features)
    labels = sel.transform(labels).tolist()[0]

    return image_features, labels


def selectgroups(image_features, parameters):
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
                    print(("{} are selected, but not in your feature files.").format(feattype))

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

    return image_features_select, labels


def main():
    parser = argparse.ArgumentParser(description='Radiomics classification')
    parser.add_argument('-ed', '--ed', metavar='ed',
                        dest='ed', type=str, required=True,
                        help='Estimator data in (HDF)')
    parser.add_argument('-tt', '--tt', metavar='tt',
                        dest='tt', type=str, required=True,
                        help='Train- and testdata in (HDF)')
    parser.add_argument('-para', '--para', metavar='para',
                        dest='para', type=str, required=True,
                        help='Parameters (JSON)')
    parser.add_argument('-out', '--out', metavar='out',
                        dest='out', type=str, required=True,
                        help='Output: fitted estimator (HDF)')
    args = parser.parse_args()

    # Convert lists into strings
    if type(args.ed) is list:
        args.ed = ''.join(args.ed)
    if type(args.tt) is list:
        args.tt = ''.join(args.tt)
    if type(args.para) is list:
        args.para = ''.join(args.para)
    if type(args.out) is list:
        args.out = ''.join(args.out)

    # Read the data
    data = pd.read_hdf(args.ed)
    traintest = pd.read_hdf(args.tt)
    with open(args.para, 'rb') as fp:
        para = json.load(fp)

    # Perform feature selection if required
    if 'FeatSel_Group' in para:
        del para['FeatSel_Group']
        feature_groups = ["histogram_features", "orientation_features",
                          "patient_features", "semantic_features",
                          "shape_features", "texture_features"]
        parameters_featsel = dict()
        for group in feature_groups:
            if group not in para:
                # Default: do use the group
                value = 1
            else:
                value = para[group]
                del para[group]

            parameters_featsel[group] = value

        X, feature_labels = selectgroups(data['X'], parameters_featsel)
    else:
        X = data['X']
        feature_labels = ['unknown']*len(X[0])
        print X, feature_labels

    if 'FeatSel_Variance' in para:
        if para['FeatSel_Variance'] == 'True':
            X, feature_labels =\
             selfeat_variance(X, feature_labels)
        del para['FeatSel_Variance']

    # Fit and score the classifier
    del para['Number']
    ret = _fit_and_score(data['base_estimator'], X, data['y'],
                         data['scorer'], traintest['train'],
                         traintest['test'], data['verbose'],
                         para, data['fit_params'], data['return_train_score'],
                         data['return_parameters'],
                         data['return_n_test_samples'],
                         data['return_times'], data['error_score'])

    source_labels = ['RET', 'feature_labels']

    source_data =\
        pd.Series([ret, feature_labels],
                  index=source_labels,
                  name='Fit and Score Output')
    source_data.to_hdf(args.out, 'RET')


if __name__ == '__main__':
    main()
