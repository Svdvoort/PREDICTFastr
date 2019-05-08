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

import SimpleITK as sitk
import sitk_helper as sitkh
import numpy as np

# CONSTANTS
TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)


def get_contour_boundary(contour):
    """Labels pixels on edge of boundary"""
    return sitk.BinaryContour(image1=contour, fullyConnected=True)


def sort_points(points):
    """Order supplied points in clockwise order"""
    x_center = np.mean(points[:, 0])
    y_center = np.mean(points[:, 1])

    angle = np.arctan2(points[:, 0] - x_center, points[:, 1] - y_center)
    sorted_index = np.argsort(angle)

    points = points[sorted_index, :]
    return points


def get_contour_boundary_points(contour):
    """Get boundary coordinates from contour"""
    # First convert the contour image to array
    contour_boundary = get_contour_boundary(contour)
    contour_boundary_array = sitkh.GetArrayFromImage(contour_boundary)
    boundary_index = np.asarray(np.nonzero(contour_boundary_array))

    boundary_points = list()
    for i in range(boundary_index.shape[1]):
        # Convert index to actual coordinates
        boundary_points.append(
            contour_boundary.TransformContinuousIndexToPhysicalPoint(
                [boundary_index[0, i], boundary_index[1, i], boundary_index[2, i]]))

    boundary_points = np.asarray(boundary_points)

    boundary_points = sort_points(boundary_points)
    # Add last point to ensure fully connected contour
    boundary_points = np.append(boundary_points, boundary_points[0:1], 0)
    return boundary_points


def get_voi_voxels(contour, image):
    """Gives back VOI (non-zero slices),  and indices of slices"""
    contour = sitk.Cast(contour, image.GetPixelID())
    # VOI is only where contour is 1
    voi_voxels = image*contour

    voi_array = sitkh.GetArrayFromImage(voi_voxels)
    voi_array_nz = voi_array[np.nonzero(voi_array)]

    new_voi = list()
    voi_slices = list()
    for i_slice in range(0, voi_array.shape[2]):
        if np.count_nonzero(voi_array[:, :, i_slice]) > 0:
            new_voi.append(voi_array[:, :, i_slice])
            voi_slices.append(i_slice)

    new_voi = np.asarray(new_voi)
    # Transpose because new_voi was constructed as list of slices, want slice
    # index last
    new_voi = np.transpose(new_voi, [1, 2, 0])
    voi_slices = np.asarray(voi_slices)
    return new_voi, voi_array_nz, voi_slices


def get_not_voi_voxels(contour, image):
    """Sets all voxels within VOI to 0"""
    contour = sitk.Cast(sitk.Not(contour), image.GetPixelID())
    voi_voxels = image*contour

    voi_array = sitkh.GetArrayFromImage(voi_voxels)
    voi_array_nz = voi_array[np.nonzero(voi_array)]

    return voi_array, voi_array_nz


def cmp(a, b):
    "Built in in python 2, but defined for python 3"
    return (a > b) - (a < b)


def turn(p, q, r):
    return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)


def _keep_left(hull, r):
    while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
            hull.pop()
    if not len(hull) or hull[-1] != r:
        hull.append(r)
    return hull


def convex_hull_points(points):
    # Graham scan
    """Returns points on convex hull of an array of points in CCW order.
    Uses Graham Scan"""
    points = sorted(points.tolist())
    l = reduce(_keep_left, points, [])
    u = reduce(_keep_left, reversed(points), [])
    total = l.extend(u[i] for i in xrange(1, len(u) - 1)) or l
    total = np.asarray(total)
    total = np.vstack({tuple(row) for row in total})
    total = sort_points(total)
    total = np.append(total, total[0:1], 0)
    return total


def convex_hull(contour):
    """Returns points on convex hull for contour, uses Graham Scan """
    points = get_contour_boundary_points(contour)
    total = convex_hull_points(points)
    return total


def local_convex_hull_points(points, N_min, N_max):
    """Find local convex points, can be used for smoothing boundary"""
    N_points = points.shape[0]
    points = points.tolist()
    points = sorted(points)
    total = list()

    if N_points < N_min:
        N_points_segment = N_points
    else:
        N_points_segment = N_min

    # Find number of points per segment such that each segment has equal number
    # of points
    N_max = np.min([N_points, N_max])
    while N_points % N_points_segment != 0 and N_points_segment < N_max:
        N_points_segment += 1

    if N_points_segment == N_max:
        N_segment = np.floor(N_points/N_points_segment)
    else:
        N_segment = N_points/N_points_segment

    N_segment = int(N_segment)

    for i in range(0, N_segment):
        if i == N_segment-1:
            temp_points = points[i*N_points_segment:-1]
        else:
            temp_points = points[i*N_points_segment:(i+1)*N_points_segment]
        l = reduce(_keep_left, temp_points, [])
        u = reduce(_keep_left, reversed(temp_points), [])
        l.extend(u[i] for i in xrange(1, len(u) - 1)) or l
        total.extend(l)
    total = np.asarray(total)
    total = np.vstack({tuple(row) for row in total})
    total = sort_points(total)
    total = np.append(total, total[0:1], 0)

    return total


def local_convex_hull(contour, N_min, N_max):
    """Find local convex hull for given contour"""
    points = get_contour_boundary_points(contour)
    total = local_convex_hull_points(points, N_min, N_max)
    return total


def get_smooth_contour(mask, N_min, N_max):
    """Find local convex hull for given contour"""
    points = get_contour_boundary_points(mask)
    total = local_convex_hull_points(points, N_min, N_max)
    return total
