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

import PREDICT.IOparser.config_io_selfeat as config_io
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def selfeat(image_features, config, labels, thresh=0.99):
    config = config_io.load_config(config)

    image_features_selected = list()
    labels_selected = list()
    if config['Featsel']['Method'] == "Manual":
        # Manual extract given features
        delrows = list()
        labels_selected = config['Featsel']['features']
        labels_selected_reordered = list()
        for idx, lab in enumerate(labels):
            if lab not in labels_selected:
                delrows.append(idx)
            else:
                labels_selected_reordered.append(lab)

        image_features_selected = np.delete(image_features, delrows, 1)

    elif config['Featsel']['Method'] == "Variance":
        # Delete features which are are the same in more than 99% of patients
        # TODO: Separate this into a different tool
        sel = VarianceThreshold(threshold=thresh*(1 - thresh))
        sel = sel.fit(image_features)
        image_features = sel.transform(image_features)
        labels = sel.transform(labels).tolist()[0]

    return image_features_selected, labels_selected_reordered


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
