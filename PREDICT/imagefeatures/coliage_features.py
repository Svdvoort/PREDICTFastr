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

import numpy as np
import itertools
from skimage.feature import greycomatrix
from scipy.stats import entropy
import SimpleITK as sitk
import os


def get_coliage_features(image, mask):
    '''
    Compute coliage features, based on:

    Prasanna et al. 2016, Co-occurrence of Local Anisotropic
    Gradient Orientations (CoLlAGe): A
    new radiomics descriptor, Nature Scientific Reports.

    '''
    # Make sure numpy uses only a single core in the svd
    os.environ["MKL_NUM_THREADS"] = '1'
    os.environ["NUMEXPR_NUM_THREADS"] = '1'
    os.environ["OMP_NUM_THREADS"] = '1'
    os.environ["OPENBLAS_NUM_THREADS"] = '1'
    os.environ["GOTO_NUM_THREADS"] = '1'

    coliage_features = list()
    coliage_labels = list()

    N_slices = image.shape[2]

    # TODO: Move to WORC
    # Should be optimized on training: from Prasanna et al. 2016
    nbins = [10, 20, 30]
    window_size = [7, 11]
    omega = 64
    wsize = 1
    dist = 2

    # Compute gradients
    gradients = np.gradient(image)
    gradient_x = gradients[0]
    gradient_y = gradients[1]

    # Determine ROI around mask
    nonzeros = np.nonzero(mask)
    xmin = np.min(nonzeros[0])
    xmax = np.max(nonzeros[0])
    ymin = np.min(nonzeros[1])
    ymax = np.max(nonzeros[1])

    # Preallocate E
    E = dict()
    for v in nbins:
        E[str(v)] = dict()

    for i_slice in range(0, N_slices):
        print("Processing slice {} / {}.").format(str(i_slice), str(N_slices))
        for v in nbins:
            print("Processing nbins {}.").format(str(v))
            for N in window_size:
                print("Processing wsize {}.").format(str(N))
                theta = np.zeros([xmax - xmin, ymax - ymin])
                for c in itertools.product(range(xmin, xmax), range(ymin, ymax)):
                    x = c[0]
                    y = c[1]

                    Fx = list()
                    Fy = list()

                    # Construct gradient vector for every pixel per cell
                    for coord_N in itertools.product(range(x - N/2, x + N/2 + 1),
                                                     range(y - N/2, y + N/2 + 1)):
                        # NOTE: Not covered in paper how to treat boundary pixels!
                        # If out of bounds, use boundary pixels
                        xc = min(max(coord_N[0], xmin), xmax)
                        yc = min(max(coord_N[1], ymin), ymax)

                        # Append gradients
                        Fx.append(gradient_x[xc, yc, i_slice])
                        Fy.append(gradient_y[xc, yc, i_slice])

                    # Store gradients in single matrix
                    F = np.zeros([N**2, 2])
                    F[:, 0] = np.asarray(Fx)
                    F[:, 1] = np.asarray(Fy)

                    # Get dominant gradient through SVM
                    U, s, V = np.linalg.svd(F)
                    theta[x - xmin, y - ymin] = np.arctan(V[1, 0]/V[0, 0])

                # Discretize theta analogue to parsanna et al.
                theta = sitk.GetImageFromArray(theta)
                theta = sitk.RescaleIntensity(theta, 0, omega - 1)
                theta = sitk.GetArrayFromImage(theta)
                theta = np.round(theta).astype(int)

                # theta = rescale_intensity(theta, out_range=(0, omega - 1))
                # theta = theta.astype(np.uint8)

                # Again, computations for each pixel
                E_temp = np.zeros([xmax - xmin, ymax - ymin])
                for c in itertools.product(range(0, theta.shape[0]), range(0, theta.shape[1])):
                    x = c[0]
                    y = c[1]

                    # If out of bounds of mask, use boundary pixels
                    xstart = max(x - N/2, 0)
                    xend = min(x + N/2, theta.shape[0])
                    ystart = max(y - N/2, 0)
                    yend = min(y + N/2, theta.shape[1])

                    theta_window = theta[xstart:xend, ystart:yend]
                    # Compute gray level co-occurence matrix
                    GLCM_matrix = greycomatrix(theta_window, [N/dist], [N/wsize],
                                               levels=omega,
                                               normed=True)

                    # Compute entropy
                    e = entropy(GLCM_matrix[:, :, 0, 0].flatten())
                    if e == -np.inf:
                        e = 1e-12
                    E_temp[x, y] = e

                # Create histogram
                # E.extend(E_temp)
                # histogram, bins = np.histogram(E, v)

                if i_slice == 0:
                    E[str(v)][str(N)] = E_temp.tolist()
                else:
                    E[str(v)][str(N)] += E_temp.tolist()

    for v in E.keys():
        E_nbins = E[v]
        for N in E_nbins.keys():
            E_ws = E_nbins[N]
            histogram, bins = np.histogram(E_ws, int(v))

            for idx, value in enumerate(histogram):
                coliage_features.append(value)
                label = ('cf_nbin{}_windows{}_bin{}').format(str(v),
                                                             str(N),
                                                             str(idx))
                coliage_labels.append(label)

    return coliage_features, coliage_labels


if __name__ == '__main__':
    get_coliage_features()
