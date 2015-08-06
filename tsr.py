# Copyright (c) 2013, Carnegie Mellon University
# All rights reserved.
# Authors: Siddhartha Srinivasa <siddh@cs.cmu.edu>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of Carnegie Mellon University nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import openravepy
import numpy
import numpy.random
import kin
import prpy.util
from numpy import pi

NANBW = numpy.ones(6)*float('nan')
EPSILON = 0.001


class TSR(object):
    """ A Task-Space-Region (TSR) represents a motion constraint. """
    def __init__(self, T0_w=None, Tw_e=None, Bw=None,
                 manip=None, bodyandlink='NULL'):
        if T0_w is None:
            T0_w = numpy.eye(4)
        if Tw_e is None:
            Tw_e = numpy.eye(4)
        if Bw is None:
            Bw = numpy.zeros((6, 2))
        pibound = (abs(Bw[3:6, :]) < pi + EPSILON)
        if pibound.any() is False:
            raise ValueError('Bw rotations must be [-pi, pi]', pibound)

        if numpy.any(Bw[0:3, 0] > Bw[0:3, 1]):
            raise ValueError('Bw translation bounds must be [min, max]', Bw)

        self.T0_w = T0_w
        self.Tw_e = Tw_e
        self.Bw = Bw

        # We will now create a continuous version of the bound to maintain:
        # 1. Bw[i,1] > Bw[i,0] which is necessary for LBFGS-B
        # 2. signed rotations, necessary for expressiveness
        Bw_cont = numpy.copy(Bw)
        for rot_idx in range(3, 6):
            if Bw_cont[rot_idx, 0] > Bw_cont[rot_idx, 1] + EPSILON:
                Bw_cont[rot_idx, 1] += 2*pi
        self._Bw_cont = Bw_cont

        if manip is None:
            self.manipindex = -1
        elif type(manip) == openravepy.openravepy_int.Robot:
            self.manipindex = manip.GetActiveManipulatorIndex()
        else:
            self.manipindex = manip
        self.bodyandlink = bodyandlink

    def to_transform(self, xyzrpy):
        """
        Converts a [x y z roll pitch yaw] into an
        end-effector transform.

        @param  xyzrpy [x y z roll pitch yaw]
        @return trans 4x4 transform
        """
        if len(xyzrpy) != 6:
            raise ValueError('xyzrpy must be of length 6')
        if not all(self.is_valid(xyzrpy)):
            raise ValueError('Invalid xyzrpy', xyzrpy)

        xyzypr = [xyzrpy[0], xyzrpy[1], xyzrpy[2],
                  xyzrpy[5], xyzrpy[4], xyzrpy[3]]
        Tw = kin.pose_to_H(kin.pose_from_xyzypr(xyzypr))
        trans = reduce(numpy.dot, [self.T0_w, Tw, self.Tw_e])
        return trans

    def to_xyzrpy(self, trans):
        """
        Converts an end-effector transform to xyzrpy values
        @param  trans  4x4 transform
        @return xyzrpy 6x1 vector of Bw values
        """
        Tw = reduce(numpy.dot, [numpy.linalg.inv(self.T0_w),
                                trans,
                                numpy.linalg.inv(self.Tw_e)])
        pose = kin.pose_from_H(Tw)
        ypr = kin.quat_to_ypr(pose[3:7])
        xyzrpy = [pose[0], pose[1], pose[2],
                  ypr[2], ypr[1], ypr[0]]
        return xyzrpy

    def is_valid(self, xyzrpy, ignoreNAN=False):
        """
        Checks if a xyzrpy is a valid sample from the TSR.
        Two main issues: dealing with roundoff issues for zero bounds and
        Wraparound for rpy.
        @param xyzrpy 6x1 vector of Bw values
        @param ignoreNAN (optional, defaults to False) ignore NaN xyzrpy
        @return a 6x1 vector of True if bound is valid and False if not
        """
        # Extract XYZ and RPY components of input and TSR.
        Bw_xyz, Bw_rpy = self.Bw[0:3, :], self.Bw[3:6, :]
        xyz, rpy = xyzrpy[0:3], xyzrpy[3:6]

        # Unwrap rpy to [-pi, pi].
        rpy = (numpy.array(rpy) + pi) % (2*pi) - pi

        # Check bounds condition on XYZ component.
        xyzcheck = [((x + EPSILON) >= Bw_xyz[i, 0]) and
                    ((x - EPSILON) <= Bw_xyz[i, 1])
                    for i, x in enumerate(xyz)]
        
        # Check bounds condition on RPY component.
        rpycheck = [False] * 3
        for i in range(0, 3):
            if (Bw_rpy[i, 0] > Bw_rpy[i, 1] + EPSILON):
                # An outer interval
                rpycheck[i] = (((rpy[i] + EPSILON) >= Bw_rpy[i, 0]) or
                               ((rpy[i] - EPSILON) <= Bw_rpy[i, 1]))
            else:
                # An inner interval
                rpycheck[i] = (((rpy[i] + EPSILON) >= Bw_rpy[i, 0]) and
                               ((rpy[i] - EPSILON) <= Bw_rpy[i, 1]))

        # Concatenate the XYZ and RPY components of the check.
        check = numpy.hstack((xyzcheck, rpycheck))

        # If ignoreNAN, components with NaN values are always OK.
        if ignoreNAN:
            check |= numpy.isnan(xyzrpy)

        return check

    def contains(self, trans):
        """
        Checks if the TSR contains the transform
        @param  trans 4x4 transform
        @return a 6x1 vector of True if bound is valid and False if not
        """
        xyzrpy = self.to_xyzrpy(trans)
        return self.is_valid(xyzrpy)

    def distance(self, trans):
        """
        Computes the Geodesic Distance from the TSR to a transform
        @param trans 4x4 transform
        @return dist Geodesic distance to TSR
        @return bwopt Closest Bw value to trans
        """
        if all(self.contains(trans)):
            return 0., self.to_xyzrpy(trans)

        import scipy.optimize

        def objective(bw):
            bwtrans = self.to_transform(bw)
            return prpy.util.GeodesicDistance(bwtrans, trans)

        bwinit = (self._Bw_cont[:, 0] + self._Bw_cont[:, 1])/2
        bwbounds = [(self._Bw_cont[i, 0], self._Bw_cont[i, 1])
                    for i in range(6)]

        bwopt, dist, info = scipy.optimize.fmin_l_bfgs_b(
                                objective, bwinit, fprime=None,
                                args=(),
                                bounds=bwbounds, approx_grad=True)
        return dist, bwopt

    def sample_xyzrpy(self, xyzrpy=NANBW):
        """
        Samples from Bw to generate an xyzrpy sample
        Can specify some values optionally as NaN.

        @param xyzrpy   (optional) a 6-vector of Bw with float('nan') for
                        dimensions to sample uniformly.
        @return         an xyzrpy sample
        """
        check = self.is_valid(xyzrpy, ignoreNAN=True)
        if not all(check):
            raise ValueError('xyzrpy must be within bounds', check)

        Bw_sample = [self._Bw_cont[i, 0] +
                     (self._Bw_cont[i, 1] - self._Bw_cont[i, 0]) *
                     numpy.random.random_sample()
                     if numpy.isnan(x) else x for i, x in enumerate(xyzrpy)]
        # Unwrap rpy to [-pi, pi]
        Bw_sample[3:6] = numpy.add(Bw_sample[3:6], pi) % (2*pi) - pi
        return Bw_sample

    def sample(self, xyzrpy=NANBW):
        """
        Samples from Bw to generate an end-effector transform.
        Can specify some Bw values optionally.

        @param xyzrpy   (optional) a 6-vector of Bw with float('nan') for
                        dimensions to sample uniformly.
        @return         4x4 transform
        """
        return self.to_transform(self.sample_xyzrpy(xyzrpy))

    def to_dict(self):
        """ Convert this TSR to a python dict. """
        return {
            'T0_w': self.T0_w.tolist(),
            'Tw_e': self.Tw_e.tolist(),
            'Bw': self.Bw.tolist(),
            'manipindex': int(self.manipindex),
            'bodyandlink': str(self.bodyandlink),
        }

    @staticmethod
    def from_dict(x):
        """ Construct a TSR from a python dict. """
        return TSR(
            T0_w=numpy.array(x['T0_w']),
            Tw_e=numpy.array(x['Tw_e']),
            Bw=numpy.array(x['Bw']),
            manip=numpy.array(x.get('manipindex', -1)),
            bodyandlink=numpy.array(x.get('bodyandlink', 'NULL'))
        )

    def to_json(self):
        """ Convert this TSR to a JSON string. """
        import json
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(x, *args, **kw_args):
        """
        Construct a TSR from a JSON string.

        This method internally forwards all arguments to `json.loads`.
        """
        import json
        x_dict = json.loads(x, *args, **kw_args)
        return TSR.from_dict(x_dict)

    def to_yaml(self):
        """ Convert this TSR to a YAML string. """
        import yaml
        return yaml.dump(self.to_dict())

    @staticmethod
    def from_yaml(x, *args, **kw_args):
        """
        Construct a TSR from a YAML string.

        This method internally forwards all arguments to `yaml.safe_load`.
        """
        import yaml
        x_dict = yaml.safe_load(x, *args, **kw_args)
        return TSR.from_dict(x_dict)


class TSRChain(object):

    def __init__(self, sample_start=False, sample_goal=False, constrain=False,
                 TSR=None, TSRs=None,
                 mimicbodyname='NULL', mimicbodyjoints=None):
        """
        A TSR chain is a combination of TSRs representing a motion constraint.

        TSR chains compose multiple TSRs and the conditions under which they
        must hold.  This class provides support for start, goal, and/or
        trajectory-wide constraints.  They can be constructed from one or more
        TSRs which must be applied together.

        @param sample_start apply constraint to start configuration sampling
        @param sample_goal apply constraint to goal configuration sampling
        @param constrain apply constraint over the whole trajectory
        @param TSR a single TSR to use in this TSR chain
        @param TSRs a list of TSRs to use in this TSR chain
        @param mimicbodyname name of associated mimicbody for this chain
        @param mimicbodyjoints 0-indexed indices of the mimicbody's joints that
                               are mimiced (MUST BE INCREASING AND CONSECUTIVE)
        """
        self.sample_start = sample_start
        self.sample_goal = sample_goal
        self.constrain = constrain
        self.mimicbodyname = mimicbodyname
        if mimicbodyjoints is None:
            self.mimicbodyjoints = []
        else:
            self.mimicbodyjoints = mimicbodyjoints
        self.TSRs = []
        if TSR is not None:
            self.append(TSR)
        if TSRs is not None:
            for tsr in TSRs:
                self.append(tsr)

    def append(self, tsr):
        self.TSRs.append(tsr)

    def to_dict(self):
        """ Construct a TSR chain from a python dict. """
        return {
            'sample_goal': self.sample_goal,
            'sample_start': self.sample_start,
            'constrain': self.constrain,
            'mimicbodyname': self.mimicbodyname,
            'mimicbodyjoints': self.mimicbodyjoints,
            'tsrs': [tsr.to_dict() for tsr in self.TSRs],
        }

    @staticmethod
    def from_dict(x):
        """ Construct a TSR chain from a python dict. """
        return TSRChain(
            sample_start=x['sample_start'],
            sample_goal=x['sample_goal'],
            constrain=x['constrain'],
            TSRs=[TSR.from_dict(tsr) for tsr in x['tsrs']],
            mimicbodyname=x['mimicbodyname'],
            mimicbodyjoints=x['mimicbodyjoints'],
        )

    def to_json(self):
        """ Convert this TSR chain to a JSON string. """
        import json
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(x, *args, **kw_args):
        """
        Construct a TSR chain from a JSON string.

        This method internally forwards all arguments to `json.loads`.
        """
        import json
        x_dict = json.loads(x, *args, **kw_args)
        return TSR.from_dict(x_dict)

    def to_yaml(self):
        """ Convert this TSR chain to a YAML string. """
        import yaml
        return yaml.dump(self.to_dict())

    @staticmethod
    def from_yaml(x, *args, **kw_args):
        """
        Construct a TSR chain from a YAML string.

        This method internally forwards all arguments to `yaml.safe_load`.
        """
        import yaml
        x_dict = yaml.safe_load(x, *args, **kw_args)
        return TSR.from_dict(x_dict)

    def is_valid(self, xyzrpy_list, ignoreNAN=False):
        """
        Checks if a xyzrpy list is a valid sample from the TSR.
        @param xyzrpy_list a list of xyzrpy values
        @param ignoreNAN (optional, defaults to False) ignore NaN xyzrpy
        @return a list of 6x1 vector of True if bound is valid and False if not
        """

        if len(xyzrpy_list) != len(self.TSRs):
            raise('Sample must be of equal length to TSR chain!')

        check = []
        for idx in range(len(self.TSRs)):
            check.append(self.TSRs[idx].is_valid(xyzrpy_list[idx], ignoreNAN))

        return check

    def to_transform(self, xyzrpy_list):
        """
        Converts a xyzrpy list into an
        end-effector transform.

        @param  a list of xyzrpy values
        @return trans 4x4 transform
        """
        check = self.is_valid(xyzrpy_list)
        for idx in range(len(self.TSRs)):
            if not all(check[idx]):
                raise ValueError('Invalid xyzrpy_list', check)

        T_sofar = self.TSRs[0].T0_w
        for idx in range(len(self.TSRs)):
            tsr_current = self.TSRs[idx]
            tsr_current.T0_w = T_sofar
            T_sofar = tsr_current.to_transform(xyzrpy_list[idx])

        return T_sofar

    def sample_xyzrpy(self, xyzrpy_list=None):
        """
        Samples from Bw to generate a list of xyzrpy samples
        Can specify some values optionally as NaN.

        @param xyzrpy_list   (optional) a list of Bw with float('nan') for
                        dimensions to sample uniformly.
        @return sample  a list of sampled xyzrpy
        """

        if xyzrpy_list is None:
            xyzrpy_list = [NANBW]*len(self.TSRs)

        sample = []
        for idx in range(len(self.TSRs)):
            sample.append(self.TSRs[idx].sample_xyzrpy(xyzrpy_list[idx]))

        return sample

    def sample(self, xyzrpy_list=None):
        """
        Samples from the Bw chain to generate an end-effector transform.
        Can specify some Bw values optionally.

        @param xyzrpy_list   (optional) a list of xyzrpy with float('nan') for
                             dimensions to sample uniformly.
        @return T0_w         4x4 transform
        """
        return self.to_transform(self.sample_xyzrpy(xyzrpy_list))

    def distance(self, trans):
        """
        Computes the Geodesic Distance from the TSR chain to a transform
        @param trans 4x4 transform
        @return dist Geodesic distance to TSR
        @return bwopt Closest Bw value to trans output as a list of xyzrpy
        """
        import scipy.optimize

        def objective(xyzrpy_list):
            xyzrpy_stack = xyzrpy_list.reshape(len(self.TSRs), 6)
            tsr_trans = self.to_transform(xyzrpy_stack)
            return prpy.util.GeodesicDistance(tsr_trans, trans)

        bwinit = []
        bwbounds = []
        for idx in range(len(self.TSRs)):
            Bw = self.TSRs[idx].Bw
            bwinit.extend((Bw[:, 0] + Bw[:, 1])/2)
            bwbounds.extend([(Bw[i, 0], Bw[i, 1]) for i in range(6)])

        bwopt, dist, info = scipy.optimize.fmin_l_bfgs_b(
                                objective, bwinit, fprime=None,
                                args=(),
                                bounds=bwbounds, approx_grad=True)
        return dist, bwopt.reshape(len(self.TSRs), 6)

    def contains(self, trans):
        """
        Checks if the TSR chain contains the transform
        @param  trans 4x4 transform
        @return       True if inside and False if not
        """
        dist, _ = self.distance(trans)
        return (abs(dist) < EPSILON)

    def to_xyzrpy(self, trans):
        """
        Converts an end-effector transform to a list of xyzrpy values
        @param  trans  4x4 transform
        @return xyzrpy_list list of xyzrpy values
        """
        _, xyzrpy_list = self.distance(trans)
        return xyzrpy_list
