#!/usr/bin/env python3
"""Sky fields for the Pasiphae survey.
"""

from abc import  ABCMeta, abstractmethod
from astropy.coordinates import Angle, SkyCoord
import json
import numpy as np
from textwrap import dedent

from utilities import close_to_edge, inside_polygon, rotate_frame

__author__ = "Sebastian Kiehlmann"
__credits__ = ["Sebastian Kiehlmann"]
__license__ = "BSD3"
__version__ = "0.1"
__maintainer__ = "Sebastian Kiehlmann"
__email__ = "skiehlmann@mail.de"
__status__ = "Production"

#==============================================================================
# CLASSES
#==============================================================================

class GuideStarSelector(metaclass=ABCMeta):
    """A class to select guide stars."""

    #--------------------------------------------------------------------------
    def __init__(self):
        """Create GuideStarSelector instance.
        """

        pass

    #--------------------------------------------------------------------------
    @abstractmethod
    def set_params(self):
        """Set science field and guide area parameters.

        Parameters
        ----------
        Depend on the specific instrument.

        Raises
        ------
        ValueError
            Raised if the grid parameters are without their allowed bounds.

        Returns
        -------
        None
        """

        # check inputs:
        # custom code goes here

        # store parameters:
        self.params = {}
        # custom code goes here, all parameters need to be stored in this dict

    #--------------------------------------------------------------------------
    def save_params(self, filename):
        """Save science field and guide area in JSON file.

        Parameters
        ----------
        filename : str
            Filename for saving the parameters.

        Returns
        -------
        None
        """

        with open(filename, mode='w') as f:
            json.dump(self.params, f, indent=4)

        print('Guide parameters saved in:', filename)

    #--------------------------------------------------------------------------
    def load_params(self, filename):
        """Load science field and guide area parameters from JSON file.

        Parameters
        ----------
        filename : str
            Filename that stores the parameters.

        Returns
        -------
        None
        """

        with open(filename, mode='r') as f:
            params = json.load(f)
            self.set_params(**params)

        print(f'Parameters loaded from {filename}.')

    #--------------------------------------------------------------------------
    def add_stars(self, ra, dec, mag):
        """Add the stars' coordinates and magnitudes, from which guide stars
        will be selected.

        Parameters
        ----------
        ra : array-like
            Right ascensions of the stars in radians.
        dec : array-like
            Declinations of the stars in radians.
        mag : array-like
            Magnitudes of the stars.
        """

        self.stars_coord = SkyCoord(ra, dec, unit='rad')
        self.stars_mag = np.asarray(mag)

#==============================================================================

class GuideStarWalopS(GuideStarSelector):
    """A class to select guide stars for WALOP-South targets."""

    #--------------------------------------------------------------------------
    def set_params(
            self, circle_radius, circle_offset, field_size, guide_area,
            limit=0, scale=1):
        """Set science field and guide area parameters.

        Parameters
        ----------
        circle_radius : float
            Radius of the full instrument area. Can be given in any unit.
            Multiplied with the `scale` factor that converts unit to radians.
        circle_offset : float
            Offset of the circle center from the science field center. Can be
            given in any unit. Multiplied with the `scale` factor that converts
            unit to radians.
        field_size : float
            Size of the science field. Can be given in any unit. Multiplied
            with the `scale` factor that converts unit to radians.
        guide_area : array-like
            Two-dimensional array (or list of lists), where each element
            defines a corner point of the guide area. Can be given in any unit.
            Multiplied with the `scale` factor that converts unit to radians.
        limit : float, optional
            Only stars located off the edge of the guide area by at least this
            limit are selected as guide stars. Can be given in any unit.
            Multiplied with the `scale` factor that converts unit to radians.
            The default is 0.
        scale : float, optional
            Scale factor that converts the above values to radians. The default
            is 1.

        Raises
        ------
        ValueError
            Raised, if `scale` is negative or zero.
            Raised, if `limit` is negative.
            Raised, if `guide_area` is not 2-dimensional.

        Returns
        -------
        None
        """

        guide_area = np.asarray(guide_area)

        # check inputs:
        if scale <= 0:
            raise ValueError("`scale` must be > 0.")
        if limit < 0:
            raise ValueError("`limit` must be >= 0.")
        if guide_area.ndim != 2:
            raise ValueError("`guide_area` must be 2-dimensional.")

        # store parameters:
        self.params = {
                'circle_radius': circle_radius,
                'circle_offset': circle_offset,
                'field_size': field_size,
                'guide_area': guide_area.tolist(),
                'limit': limit,
                'scale': scale}
        self.circle_radius = Angle(circle_radius*scale, unit='rad')
        self.circle_offset = Angle(circle_offset*scale, unit='rad')
        self.field_size = Angle(field_size*scale, unit='rad')
        self.guide_area = Angle(guide_area*scale, unit='rad')
        self.limit = Angle(limit*scale, unit='rad')

    #--------------------------------------------------------------------------
    def select(self, field_ra, field_dec):
        # TODO

        # select closest stars
        radius = self.circle_radius - self.limit
        circle_center = SkyCoord(
                field_ra + self.circle_offset.rad,
                field_dec + self.circle_offset.rad,
                unit='rad')
        sel_circle = self.stars_coord.separation(circle_center) < radius
        i_circle = np.nonzero(sel_circle)[0]
        candidates_coord = self.stars_coord[sel_circle]

        # rotate coordinate frame:
        field_center = SkyCoord(field_ra, field_dec, unit='rad')
        ra_rot, dec_rot = rotate_frame(
                candidates_coord.ra.rad, candidates_coord.dec.rad, field_center
                )
        n = ra_rot.shape[0]

        # select candidates within guide area:
        sel_guide = np.zeros(n, dtype=bool)

        for i, point in enumerate(zip(ra_rot, dec_rot)):
            sel_guide[i] = inside_polygon(point, self.guide_area)

        i_guide = i_circle[sel_guide]
        candidates_coord = candidates_coord[sel_guide]
        n = i_guide.shape[0]

        # select candidates far enough from the guide area edges:
        sel_edge = np.zeros(n, dtype=bool)

        for i, point in enumerate(zip(
                    candidates_coord.ra.rad, candidates_coord.dec.rad)):
            sel_edge[i] = close_to_edge(
                    point, self.guide_area, self.limit.rad)

        i_edge = i_guide[sel_edge]
        i_guide = i_guide[~sel_edge]
        candidates_coord = candidates_coord[~sel_edge]

        return i_guide, i_edge, i_circle

#==============================================================================
