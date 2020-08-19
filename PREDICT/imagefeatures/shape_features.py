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
from skimage.measure import label
import SimpleITK as sitk
import PREDICT.helpers.contour_functions as cf
import PREDICT.helpers.sitk_helper as sitkh

# CONSTANTS
N_min_smooth = 10
N_max_smooth = 40


def get_shape_features(mask, metadata=None, mode='2D'):
    '''
    Compute all shape features on a mask. Returns two lists: the feature values
    and the feature labels.
    '''
    if mode == '3D':
        features, labels = get_shape_features_3D(mask, metadata)
        labels = [l + '_3D' for l in labels]
    else:
        if len(mask.GetSize()) == 2:
            features, labels = get_shape_features_1D(mask, metadata)
            labels = [l + '_1D' for l in labels]
        else:
            features, labels = get_shape_features_2D(mask, metadata)
            labels = [l + '_2D' for l in labels]

    return features, labels


def get_shape_features_single_slice(maskArray, metadata=None):
    """
    Get shape features of a single 2D slice.

    Can contain multiple blobs.
    """
    # Initialize lists of features which we compute per blob
    convexity_temp = list()
    rad_dist_temp = list()
    roughness_temp = list()
    roughness_avg_temp = list()
    cvar_temp = list()
    prax_temp = list()
    evar_temp = list()
    solidity_temp = list()
    compactness_temp = list()
    area = 0

    # Loop over first blob, which has value 1, till last blob
    for nblob in range(0 + 1, np.max(maskArray) + 2):
        blobimage = np.zeros(maskArray.shape)
        blobimage[maskArray == nblob] = 1

        if np.sum(blobimage) == 0:
            # No points in volume, therefore we ignore it.
            continue

        blobimage = blobimage.astype(np.uint8)
        blobimage = sitk.GetImageFromArray(blobimage)

        # Iterate over all blobs in a slice
        boundary_points = cf.get_smooth_contour(blobimage,
                                                N_min_smooth,
                                                N_max_smooth)
        if boundary_points.shape[0] <= 3:
            # Only 1 or 2 points in volume, which means it's not really a
            # volume, therefore we ignore it.
            continue

        rad_dist_i, _ = compute_radial_distance(
            boundary_points)
        rad_dist_temp.extend(rad_dist_i)
        perimeter = compute_perimeter(boundary_points)

        area += compute_area(boundary_points)
        compactness_temp.append(compute_compactness(boundary_points))
        roughness_i, roughness_avg = compute_roughness(
            boundary_points, rad_dist_i)
        roughness_avg_temp.append(roughness_avg)
        roughness_temp.extend(roughness_i)

        cvar_temp.append(compute_cvar(boundary_points))
        prax_temp.append(compute_prax(boundary_points))
        evar_temp.append(compute_evar(boundary_points))

        # TODO: Move computing convexity into esf
        convex_hull = cf.convex_hull(blobimage)
        convexity_temp.append(compute_perimeter(convex_hull / perimeter))

        solidity_temp.append(compute_area(convex_hull)/area)

    # Take averages of some features
    convexity = np.mean(convexity_temp)
    rad_dist_avg = np.mean(np.asarray(rad_dist_temp))
    rad_dist_std = np.std(np.asarray(rad_dist_temp))
    roughness_avg = np.mean(np.asarray(roughness_avg_temp))
    roughness_std = np.std(np.asarray(roughness_temp))
    cvar = np.mean(cvar_temp)
    prax = np.mean(prax_temp)
    evar = np.mean(evar_temp)
    solidity = np.mean(solidity_temp)
    compactness = np.mean(compactness_temp)

    return convexity, area, rad_dist_avg, rad_dist_std,\
        roughness_avg, roughness_std, cvar, prax, evar, solidity,\
        compactness


def get_shape_features_3D(mask_ITKim, metadata=None):
    mask = sitk.GetArrayFromImage(mask_ITKim)

    # Pre-allocation
    perimeter = list()
    convexity = list()
    area = list()
    rad_dist_avg = list()
    rad_dist_std = list()
    roughness_avg = list()
    roughness_std = list()
    cvar = list()
    prax = list()
    evar = list()
    solidity = list()
    compactness = list()

    rad_dist = list()
    rad_dist_norm = list()
    roughness = list()

    # Now calculate some of the edge shape features
    # TODO: Adapt to allow for multiple slices at once
    labeledImage = label(mask, connectivity=3)
    for nblob in range(1, np.max(labeledImage)):
        # Iterate over all slices
        blobimage = np.zeros(mask.shape)
        blobimage[mask == nblob] = 1
        blobimage = blobimage.astype(np.uint8)
        if np.sum(blobimage) <= 3:
            # Only 1 or 2 points in volume, which means it's not really a
            # volume, therefore we ignore it.
            continue

        blobimage = sitk.GetImageFromArray(blobimage)

        boundary_points = cf.get_smooth_contour(blobimage,
                                                N_min_smooth,
                                                N_max_smooth)
        if boundary_points.shape[0] <= 3:
            # Only 1 or 2 points in volume, which means it's not really a
            # volume, therefore we ignore it.
            continue
        rad_dist_i, rad_dist_norm_i = compute_radial_distance(
            boundary_points)
        rad_dist.append(rad_dist_i)
        rad_dist_norm.append(rad_dist_norm_i)
        perimeter.append(compute_perimeter(boundary_points))

        area.append(compute_area(boundary_points))
        compactness.append(compute_compactness(boundary_points))
        roughness_i, roughness_avg_temp = compute_roughness(
            boundary_points, rad_dist_i)
        roughness_avg.append(roughness_avg_temp)
        roughness.append(roughness_i)

        cvar.append(compute_cvar(boundary_points))
        prax.append(compute_prax(boundary_points))
        evar.append(compute_evar(boundary_points))

        # TODO: Move computing convexity into esf
        convex_hull = cf.convex_hull(blobimage)
        convexity.append(compute_perimeter(convex_hull)
            / perimeter[-1])

        solidity.append(compute_area(convex_hull)/area[-1])
        rad_dist_avg.append(np.mean(np.asarray(rad_dist_i)))
        rad_dist_std.append(np.std(np.asarray(rad_dist_i)))
        roughness_std.append(np.std(np.asarray(roughness_i)))

    compactness_avg = np.mean(compactness)
    compactness_std = np.std(compactness)
    compactness_std = np.std(compactness)
    compactness_std = np.std(compactness)
    convexity_avg = np.mean(convexity)
    convexity_std = np.std(convexity)
    rad_dist_avg = np.mean(rad_dist_avg)
    rad_dist_std = np.mean(rad_dist_std)
    roughness_avg = np.mean(roughness_avg)
    roughness_std = np.mean(roughness_std)
    cvar_avg = np.mean(cvar)
    cvar_std = np.std(cvar)
    prax_avg = np.mean(prax)
    prax_std = np.std(prax)
    evar_avg = np.mean(evar)
    evar_std = np.std(evar)
    solidity_avg = np.mean(solidity)
    solidity_std = np.std(solidity)

    shape_labels = ['sf_compactness_avg', 'sf_compactness_std', 'sf_rad_dist_avg',
                    'sf_rad_dist_std', 'sf_roughness_avg', 'sf_roughness_std',
                    'sf_convexity_avg', 'sf_convexity_std', 'sf_cvar_avg', 'sf_cvar_std',
                    'sf_prax_avg', 'sf_prax_std', 'sf_evar_avg', 'sf_evar_std',
                    'sf_solidity_avg', 'sf_solidity_std']

    shape_features = [compactness_avg, compactness_std, rad_dist_avg,
                      rad_dist_std, roughness_avg, roughness_std,
                      convexity_avg, convexity_std, cvar_avg, cvar_std,
                      prax_avg, prax_std, evar_avg, evar_std, solidity_avg,
                      solidity_std]

    if metadata is not None:
        if (0x18, 0x50) in metadata.keys():
            # import dicom as pydicom
            # metadata = pydicom.read_file(metadata)
            pixel_spacing = metadata[0x28, 0x30].value
            slice_thickness = int(metadata[0x18, 0x50].value)
            voxel_volume = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
            volume = np.sum(mask) * voxel_volume
            shape_labels.append('sf_volume')
            shape_features.append(volume)

    if voxel_volume is not None:
        # Check if we can use the pixel information from the Nifti
        if hasattr(mask_ITKim, 'GetSpacing'):
            spacing = mask_ITKim.GetSpacing()
            voxel_volume = spacing[0] * spacing[1] * spacing[2]
            volume = np.sum(mask) * voxel_volume
            shape_labels.append('sf_volume')
            shape_features.append(volume)

    return shape_features, shape_labels


def get_shape_features_2D(mask_ITKim, metadata=None):
    # Pre-allocation
    convexity = list()
    area = list()
    rad_dist_avg = list()
    rad_dist_std = list()
    roughness_avg = list()
    roughness_std = list()
    cvar = list()
    prax = list()
    evar = list()
    solidity = list()
    compactness = list()

    # Determine Voxel Size
    voxel_volume = voxel_area = None
    if metadata is not None:
        if (0x18, 0x50) in metadata.keys():
            # import dicom as pydicom
            # metadata = pydicom.read_file(metadata)
            pixel_spacing = metadata[0x28, 0x30].value
            slice_thickness = int(metadata[0x18, 0x50].value)
            voxel_volume = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
            voxel_area = pixel_spacing[0] * pixel_spacing[1]

    if voxel_volume is None:
        # Check if we can use the pixel information from the Nifti
        if hasattr(mask_ITKim, 'GetSpacing'):
            spacing = mask_ITKim.GetSpacing()
            voxel_volume = spacing[0] * spacing[1] * spacing[2]
            voxel_area = spacing[0] * spacing[1]

    # Now calculate some of the edge shape features
    # NOTE: Due to conversion to array, first and third axis are switched
    mask = sitkh.GetArrayFromImage(mask_ITKim)
    N_mask_slices = mask.shape[2]
    mask = label(mask, connectivity=3)
    for i_slice in range(0, N_mask_slices):
        # Iterate over all slices
        slicie = mask[:, :, i_slice]
        convexity_temp, area_temp, rad_dist_avg_temp, rad_dist_std_temp,\
            roughness_avg_temp, roughness_std_temp, cvar_temp, prax_temp,\
            evar_temp, solidity_temp, compactness_temp =\
            get_shape_features_single_slice(slicie, metadata)
        convexity.append(convexity_temp)
        if voxel_area is not None:
            area_temp *= voxel_area
        area.append(area_temp)
        rad_dist_avg.append(rad_dist_avg_temp)
        rad_dist_std.append(rad_dist_std_temp)
        roughness_avg.append(roughness_avg_temp)
        roughness_std.append(roughness_std_temp)
        cvar.append(cvar_temp)
        prax.append(prax_temp)
        evar.append(evar_temp)
        solidity.append(solidity_temp)
        compactness.append(compactness_temp)

    compactness_avg = np.mean(compactness)
    compactness_std = np.std(compactness)
    convexity_avg = np.mean(convexity)
    convexity_std = np.std(convexity)
    rad_dist_avg = np.mean(rad_dist_avg)
    rad_dist_std = np.mean(rad_dist_std)
    roughness_avg = np.mean(roughness_avg)
    roughness_std = np.mean(roughness_std)
    cvar_avg = np.mean(cvar)
    cvar_std = np.std(cvar)
    prax_avg = np.mean(prax)
    prax_std = np.std(prax)
    evar_avg = np.mean(evar)
    evar_std = np.std(evar)
    solidity_avg = np.mean(solidity)
    solidity_std = np.std(solidity)
    area_avg = np.mean(area)
    area_std = np.std(area)
    area_min = np.min(area)
    area_max = np.max(area)

    shape_labels = ['sf_compactness_avg', 'sf_compactness_std', 'sf_rad_dist_avg',
                    'sf_rad_dist_std', 'sf_roughness_avg', 'sf_roughness_std',
                    'sf_convexity_avg', 'sf_convexity_std', 'sf_cvar_avg', 'sf_cvar_std',
                    'sf_prax_avg', 'sf_prax_std', 'sf_evar_avg', 'sf_evar_std',
                    'sf_solidity_avg', 'sf_solidity_std', 'sf_area_avg',
                    'sf_area_max', 'sf_area_min', 'sf_area_std']

    shape_features = [compactness_avg, compactness_std, rad_dist_avg,
                      rad_dist_std, roughness_avg, roughness_std,
                      convexity_avg, convexity_std, cvar_avg, cvar_std,
                      prax_avg, prax_std, evar_avg, evar_std, solidity_avg,
                      solidity_std, area_avg, area_max, area_min, area_std]

    if voxel_volume is not None:
        volume = np.sum(mask) * voxel_volume
        shape_labels.append('sf_volume')
        shape_features.append(volume)

    return shape_features, shape_labels


def get_shape_features_1D(mask_ITKim, metadata=None):
    # Determine Voxel Size
    voxel_area = None
    if metadata is not None:
        pixel_spacing = metadata[0x28, 0x30].value
        voxel_area = pixel_spacing[0] * pixel_spacing[1]
    else:
        # Check if we can use the pixel information from the Nifti
        if hasattr(mask_ITKim, 'GetSpacing'):
            spacing = mask_ITKim.GetSpacing()
            voxel_area = spacing[0] * spacing[1]

    # Now calculate some of the edge shape features
    # NOTE: Due to conversion to array, first and third axis are switched
    mask = sitkh.GetArrayFromImage(mask_ITKim)
    mask = label(mask, connectivity=2)
    convexity, area, rad_dist_avg, rad_dist_std,\
        roughness_avg, roughness_std, cvar, prax,\
        evar, solidity, compactness =\
        get_shape_features_single_slice(mask, metadata)

    if voxel_area is not None:
        area *= area

    shape_labels = ['sf_compactness', 'sf_rad_dist_avg', 'sf_rad_dist_std',
                    'sf_roughness_avg', 'sf_roughness_std',
                    'sf_convexity', 'sf_cvar',
                    'sf_prax', 'sf_evar',
                    'sf_solidity', 'sf_area']

    shape_features = [compactness, rad_dist_avg, rad_dist_std,
                      roughness_avg, roughness_std,
                      convexity, cvar,
                      prax, evar,
                      solidity, area]

    return shape_features, shape_labels


def get_center(points):
    """Computes the center of the given boundary points"""
    x_center = np.mean(points[:, 0])
    y_center = np.mean(points[:, 1])
    if points.shape[1] == 3:
        # 3D: Also give z
        z_center = np.mean(points[:, 2])
        return x_center, y_center, z_center
    else:
        return x_center, y_center


def compute_dist_to_center(points):
    """Computes the distance to the center for the given boundary points"""
    center = get_center(points)

    dist_to_center = points - center
    return dist_to_center


def compute_abs_dist_to_center(points):
    """Computes the absolute distance to center for given boundary points"""

    dist_center = compute_dist_to_center(points)
    abs_dist_to_center = np.sqrt(dist_center[:, 0]**2 + dist_center[:, 1]**2)

    return abs_dist_to_center


def compute_perimeter(points):
    """Computes the perimeter of the given boundary points"""
    xdiff = np.diff(points[:, 0])
    ydiff = np.diff(points[:, 1])

    perimeter = np.sum(np.sqrt(xdiff**2 + ydiff**2))

    return perimeter


def compute_area(points):
    """Computes the area of the given boundary points using shoelace formula"""
    x = points[:, 0]
    y = points[:, 1]

    area = 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return area


def compute_compactness(points):
    """Computes compactness of the given boundary points, 1 for circle"""

    perimeter = compute_perimeter(points)
    area = compute_area(points)

    compactness = 4*np.pi*(area/perimeter**2)

    return compactness


def compute_radial_distance(points):
    """
    Computes the radial distance for the given boundary points, according to
    Xu et al 2012, "A comprehensive descriptor of shape"

    """
    dist_center = compute_dist_to_center(points)
    rad_dist = np.sqrt(dist_center[:, 0]**2 + dist_center[:, 1]**2)
    rad_dist_norm = rad_dist/np.amax(rad_dist)

    return rad_dist, rad_dist_norm


def compute_roughness(points, rad_distance=None, min_points=3, max_points=15):
    """
    compute_roughness computes the roughness according to "Xu et al. 2012,
    A comprehensive descriptor of shape"

    Args:
        points ([Nx2] numpy array): array of boundary points

    Kwargs:
        rad_distance (numpy array): Radial distance if already computed
                                    [default: None]
        min_points (int): Minimum number of points in a segment
                          [default: 3]
        max_points (int): Maximum number of points in a segment
                          [default: 15]

    Returns:
        roughness (numpy array): The roughness in the different segments
        roughness_avg (float): The average roughness

    """
    if rad_distance is None:
        rad_distance = compute_radial_distance(points)

    N_points = points.shape[0]

    # Find the number of points in a segment, by looking for number that will
    # perfectly divide the boundary into equal segements
    N_points_segment = min_points
    while N_points % N_points_segment != 0 and N_points_segment < max_points:
        N_points_segment += 1

    if N_points_segment == max_points:
        # Not perfectly divisble, so round down
        N_segments = np.floor(N_points/N_points_segment)
    else:
        N_segments = N_points/N_points_segment

    N_segments = int(N_segments)

    roughness = np.zeros([N_segments, 1])

    for i_segment in range(0, N_segments):
        if i_segment == N_segments - 1:
            # If the number of segments is not a perfect fit, the last segment
            # gets all the leftover points
            cur_segment = range(i_segment*N_points_segment, N_points)
        else:
            cur_segment = range(i_segment*N_points_segment,
                                (i_segment+1)*N_points_segment)

        roughness[i_segment] = np.sum(np.abs(np.diff(
                                                rad_distance[cur_segment])))

    roughness_avg = 1.0*N_points_segment/N_points*np.sum(roughness)

    return roughness, roughness_avg


def compute_mean_radius(points):
    """
    Computes mean radius for giving boundary points, according to Peura et al.
    1997, "Efficiency of Simple Shape Descriptors"

    """
    abs_dist_center = compute_abs_dist_to_center(points)
    mean_radius = 1.0*np.sum(abs_dist_center)/points.shape[0]

    return mean_radius


def compute_cvar(points):
    """
    Computes circular variance for giving boundary points, according to
    Peura et al. 1997, "Efficiency of Simple Shape Descriptors"

    """
    abs_dist_center = compute_abs_dist_to_center(points)

    mean_radius = compute_mean_radius(points)

    cvar = 1.0/(points.shape[0]*mean_radius**2)*np.sum((abs_dist_center -
                                                        mean_radius)**2)

    return cvar


def compute_covariance_matrix(points):
    """
    Computes covariance matrix for giving boundary points, according to
    Peura et al. 1997, "Efficiency of Simple Shape Descriptors"

    """
    dist_to_center = compute_dist_to_center(points)

    covariance_matrix = list()
    for i_point in dist_to_center:
        covariance_matrix.append(np.outer(i_point, i_point))

    covariance_matrix = np.asarray(covariance_matrix)
    covariance_matrix = 1.0*np.sum(covariance_matrix, 0)/points.shape[0]

    return covariance_matrix


def compute_prax(points):
    """
    Computes ratio of principal axes for giving boundary points, according to
    Peura et al. 1997, "Efficiency of Simple Shape Descriptors"

    """
    covariance_matrix = compute_covariance_matrix(points)

    c_xx = covariance_matrix[0][0]
    c_yy = covariance_matrix[1][1]
    c_xy = covariance_matrix[0][1]

    first_term = c_xx + c_yy
    second_term = np.sqrt((c_xx + c_yy)**2 - 4*(c_xx*c_yy - c_xy**2))

    prax = 1.0*(first_term - second_term)/(first_term + second_term)

    return prax


def compute_evar(points):
    """
    Computes eliptic variance for giving boundary points, according to
    Peura et al. 1997, "Efficiency of Simple Shape Descriptors"

    """
    covariance_matrix = compute_covariance_matrix(points)
    dist_to_center = compute_dist_to_center(points)

    if covariance_matrix[0][0] == 0 or covariance_matrix[1][1] == 0:
        # It isn't a well-defined contour
        return 0

    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    root_term = list()

    for i_point in dist_to_center:
        first_product = np.dot(i_point, inv_covariance_matrix)
        second_product = np.dot(first_product, i_point)
        root_term.append(np.sqrt(second_product))

    root_term = np.asarray(root_term)

    mu_rc = 1.0/points.shape[0]*np.sum(root_term)

    evar = 1.0/(points.shape[0]*mu_rc)*np.sum((root_term - mu_rc)**2)

    return evar
