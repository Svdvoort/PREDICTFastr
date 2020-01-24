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


def data_regularize(data, type="spheric", divs = 10):
    limits = np.array([
        [min(data[:,0]), max(data[:,0])],
        [min(data[:,1]), max(data[:,1])],
        [min(data[:,2]), max(data[:,2])]])

    regularized = []

    if type=="cubic":

        X = np.linspace(*limits[0], num = divs)
        Y = np.linspace(*limits[1], num = divs)
        Z = np.linspace(*limits[2], num = divs)

        for i in range(divs-1):
            for j in range(divs-1):
                for k in range(divs-1):
                    points_in_sector = []
                    for point in data:
                        if (point[0] >= X[i] and point[0] < X[i+1] and
                            point[1] >= Y[j] and point[1] < Y[j+1] and
                            point[2] >= Z[k] and point[2] < Z[k+1]) :

                            points_in_sector.append(point)
                    if len(points_in_sector) > 0:
                        regularized.append(np.mean(np.array(points_in_sector), axis=0))

    elif type=="spheric" :
        divs_u = divs
        divs_v = divs * 2

        center = np.array([
            0.5 * (limits[0,0] + limits[0,1]),
            0.5 * (limits[1,0] + limits[1,1]),
            0.5 * (limits[2,0] + limits[2,1])])
        d_c = data - center

        #spherical coordinates around center
        r_s = np.sqrt(d_c[:,0]**2. + d_c[:,1]**2. + d_c[:,2]**2.)
        d_s = np.array([
            r_s,
            np.arccos(d_c[:,2] / r_s),
            np.arctan2(d_c[:,1], d_c[:,0])]).T

        u = np.linspace(0, np.pi, num = divs_u)
        v = np.linspace(-np.pi, np.pi, num = divs_v)

        for i in range(divs_u - 1):
            for j in range(divs_v - 1):
                points_in_sector = []
                for k , point in enumerate(d_s):
                    if (point[1] >= u[i] and point[1] < u[i+1] and
                        point[2] >= v[j] and point[2] < v[j+1]) :

                        points_in_sector.append(data[k])

                if len(points_in_sector) > 0:
                    regularized.append(np.mean(np.array(points_in_sector), axis=0))

    return np.array(regularized)


# https://github.com/minillinim/ellipsoid
def ellipsoid_plot(center, radii, rotation, ax, plotAxes=False, cageColor='b', cageAlpha=0.2):
    """Plot an ellipsoid"""

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    if plotAxes:
        # make some purdy axes
        axes = np.array([[radii[0],0.0,0.0],
                         [0.0,radii[1],0.0],
                         [0.0,0.0,radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)


        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cageColor)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)


# http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
# for arbitrary axes
def ellipsoid_fit(X):
    x=X[:,0]
    y=X[:,1]
    if X.shape[1] == 2:
        z = np.ones(y.shape)
    else:
        z=X[:,2]
    D = np.array([x*x,
                 y*y,
                 z*z,
                 2 * x*y,
                 2 * x*z,
                 2 * y*z,
                 2 * x,
                 2 * y,
                 2 * z])
    DT = D.conj().T
    v = np.linalg.solve( D.dot(DT), D.dot( np.ones( np.size(x) ) ) )
    A = np.array(  [[v[0], v[3], v[4], v[6]],
                    [v[3], v[1], v[5], v[7]],
                    [v[4], v[5], v[2], v[8]],
                    [v[6], v[7], v[8], -1]])

    center = np.linalg.solve(- A[:3,:3], [[v[6]],[v[7]],[v[8]]])
    T = np.eye(4)
    T[3,:3] = center.T
    R = T.dot(A).dot(T.conj().T)
    evals, evecs = np.linalg.eig(R[:3,:3] / -R[3,3])
    radii = np.sqrt(1. / evals)

    return center, radii, evecs, v


def ellipsoid_fit_2D(X):
    # From https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
    x = X[:, 0]
    y = X[:, 1]

    # Convert to 2D matrices
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)

    # Formulate and solve the least squares problem ||Ax - b ||^2
    A = np.hstack([x**2, x * y, y**2, x, y])
    b = np.ones_like(x)
    solution = np.linalg.lstsq(A, b)[0].squeeze()
    return solution
