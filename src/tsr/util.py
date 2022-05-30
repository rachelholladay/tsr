# Geodesic Distance
# wrap_to_interval
# GetManipulatorIndex

import logging
import math
import numpy
import scipy.misc
import scipy.optimize
import threading
import time
import warnings



def wrap_to_interval(angles, lower=-numpy.pi):
    """
    Wraps an angle into a semi-closed interval of width 2*pi.

    By default, this interval is `[-pi, pi)`.  However, the lower bound of the
    interval can be specified to wrap to the interval `[lower, lower + 2*pi)`.
    If `lower` is an array the same length as angles, the bounds will be
    applied element-wise to each angle in `angles`.

    See: http://stackoverflow.com/a/32266181

    @param angles an angle or 1D array of angles to wrap
    @type  angles float or numpy.array
    @param lower optional lower bound on wrapping interval
    @type  lower float or numpy.array
    """
    if any(numpy.isnan(angles)):
        return angles
    return (angles - lower) % (2 * numpy.pi) + lower

#TODO this is copied from pb_robot/transformations.py. 
def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True
    """
    R = numpy.array(matrix, dtype=numpy.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = numpy.linalg.eig(R33.T)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-1)[0] #XXX was < 1e-8
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = numpy.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = numpy.linalg.eig(R)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-1)[0] #XXX was < 1e-8 
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = numpy.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (numpy.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point


def GeodesicError(t1, t2):
    """
    Computes the error in global coordinates between two transforms.

    @param t1 current transform
    @param t2 goal transform
    @return a 4-vector of [dx, dy, dz, solid angle]
    """
    trel = numpy.dot(numpy.linalg.inv(t1), t2)
    trans = numpy.dot(t1[0:3, 0:3], trel[0:3, 3])
    #angle = numpy.dot(t1[0:3, 0:3], openravepy.axisAngleFromRotationMatrix(trel[0:3, 0:3]))
    try:
        omega, _, _ = rotation_from_matrix(trel)
    except:
        omega, _, _ = rotation_from_matrix(-trel)
    angle = numpy.linalg.norm(omega) 
    #angle,direction,point = (trel)
    return numpy.hstack((trans, angle))



def GeodesicDistance(t1, t2, r=1.0):
    """
    Computes the geodesic distance between two transforms

    @param t1 current transform
    @param t2 goal transform
    @param r in units of meters/radians converts radians to meters
    """
    error = GeodesicError(t1, t2)
    error[3] = r * error[3]
    return numpy.linalg.norm(error)
