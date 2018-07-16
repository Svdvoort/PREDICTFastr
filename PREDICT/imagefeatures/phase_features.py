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
import PREDICT.imagefeatures.histogram_features as hf
import PREDICT.helpers.image_helper as ih
# import imagefeatures.histogram_features as hf
# import helpers.image_helper as ih
import numpy as np
import phasepack as pp

N_BINS = 50


def get_phase_features(image, mask, parameters=dict()):
    # Alternatively, one could use the pxehancement function
    if "minwavelength" in parameters.keys():
        minwavelength = parameters["minwavelength"]
    else:
        minwavelength = [3]

    if "nscale" in parameters.keys():
        nscale = parameters["nscale"]
    else:
        nscale = [5]

    # Make a dummy
    phase_features = list()
    phase_labels = list()

    for i_index, (i_wl, i_sc) in enumerate(zip(minwavelength, nscale)):
        monogenic_image = np.zeros(image.shape)
        phasecon_image = np.zeros(image.shape)
        phasesym_image = np.zeros(image.shape)
        for i_slice in range(0, image.shape[2]):
            # Phase Congruency using Monogenic signal
            M, ori, ft, T = pp.phasecongmono(image[:, :, i_slice],
                                             nscale=i_sc,
                                             minWaveLength=i_wl,
                                             mult=2.1,
                                             sigmaOnf=0.55, k=2.,
                                             cutOff=0.5, g=10.,
                                             noiseMethod=-1,
                                             deviationGain=1.5)
            monogenic_image[:, :, i_slice] = ori
            phasecon_image[:, :, i_slice] = M

            # Phase Symmetry using Monogenic signal
            phaseSym, totalEnergy, T = pp.phasesymmono(image[:, :, i_slice],
                                                       nscale=i_sc,
                                                       minWaveLength=i_wl,
                                                       mult=2.1,
                                                       sigmaOnf=0.55, k=2.,
                                                       noiseMethod=-1)
            phasesym_image[:, :, i_slice] = phaseSym

        # Get histogram features
        masked_voxels = ih.get_masked_voxels(monogenic_image, mask)
        masked_voxels = replacenan(masked_voxels)
        histogram_features, histogram_labels = hf.get_histogram_features(masked_voxels, N_BINS)
        histogram_labels = [l.replace('hf_', 'phasef_monogenic_') for l in histogram_labels]
        phase_features.extend(histogram_features)
        final_feature_names = [feature_name + '_WL' + str(i_wl) + '_N' + str(i_sc) for feature_name in histogram_labels]
        phase_labels.extend(final_feature_names)

        masked_voxels = ih.get_masked_voxels(phasecon_image, mask)
        masked_voxels = replacenan(masked_voxels)
        histogram_features, histogram_labels = hf.get_histogram_features(masked_voxels, N_BINS)
        histogram_labels = [l.replace('hf_', 'phasef_phasecong_') for l in histogram_labels]
        phase_features.extend(histogram_features)
        final_feature_names = [feature_name + '_WL' + str(i_wl) + '_N' + str(i_sc) for feature_name in histogram_labels]
        phase_labels.extend(final_feature_names)

        masked_voxels = ih.get_masked_voxels(phasesym_image, mask)
        masked_voxels = replacenan(masked_voxels)
        histogram_features, histogram_labels = hf.get_histogram_features(masked_voxels, N_BINS)
        histogram_labels = [l.replace('hf_', 'phasef_phasesym_') for l in histogram_labels]
        phase_features.extend(histogram_features)
        final_feature_names = [feature_name + '_WL' + str(i_wl) + '_N' + str(i_sc) for feature_name in histogram_labels]
        phase_labels.extend(final_feature_names)

    return phase_features, phase_labels


def replacenan(x):
    # First, replace the NaNs:
    X_notnan = x[:]
    for fnum, f in enumerate(x):
        if np.isnan(f):
            print("[PREDICT WARNING] NaN found in phase features. Replacing with zero.")
            X_notnan[fnum] = 0

    return X_notnan
