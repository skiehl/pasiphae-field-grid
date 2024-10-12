#!/usr/bin/env python3
"""Sky fields for the Pasiphae survey.
"""

from abc import  ABCMeta, abstractmethod
from astropy.coordinates import Angle, SkyCoord
import json
import matplotlib.pyplot as plt
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
    def _select(self, field_ra, field_dec, return_coord=False):
        """Select guide stars for one field.

        Parameters
        ----------
        field_ra : float
            Field center right ascension in radians.
        field_dec : float
            Field center declination in radians.
        return_coord : bool, optional
            If True, coordinates of the selected guide stars, of stars too
            close to the edge of the guider area, and stars within the
            instrument area are returned in addition to the indices of the
            selected stars. Otherwise, only the indices of the selected stars
            are returned.

        Returns
        -------
        i_guide : numpy.ndarray
            Indices of the stars in the guide area.
        coord_rot_guide : astropy.coordinates.SkyCoord
            Coordinates of the selected guide stars. Only returned if
            `return_coord=True`.
        coord_rot_edge : astropy.coordinates.SkyCoord
            Coordinates of the stars in the guide area, but too close to the
            edge. Only returned if `return_coord=True`.
        coord_rot_circle : astropy.coordinates.SkyCoord
            Coordinates of the stars in the instrument area. Only returned if
            `return_coord=True`.
        """

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
        n = i_guide.shape[0]

        # select candidates far enough from the guide area edges:
        sel_edge = np.zeros(n, dtype=bool)

        for i, point in enumerate(zip(ra_rot[sel_guide], dec_rot[sel_guide])):
            sel_edge[i] = close_to_edge(
                    point, self.guide_area, self.limit.rad)

        i_guide = i_guide[~sel_edge]

        if return_coord:
            coord_rot_circle = SkyCoord(ra_rot, dec_rot, unit='rad')
            coord_rot_edge = coord_rot_circle[sel_guide][sel_edge]
            coord_rot_guide = coord_rot_circle[sel_guide][~sel_edge]

            return i_guide, coord_rot_guide, coord_rot_edge, coord_rot_circle

        else:
            return i_guide

    #--------------------------------------------------------------------------
    def _iter_grid(self, fieldgrid):
        # TODO

        field_ras, field_decs = fieldgrid.get_center_coords()
        n = len(fieldgrid)
        print('Iterate through field grid..')

        for i, (field_ra, field_dec) in enumerate(zip(field_ras, field_decs)):
            print(f'\rField {i} of {n} ({i/n*100:.1f}%)..', end='')

            i_guide, __, __, __ = self._select(field_ra, field_dec)

            # TODO: further processing

        print('\rdone.                             ')

    #--------------------------------------------------------------------------
    def select(self, field_ra=None, field_dec=None, fieldgrid=None):
        # TODO

        if field_ra is not None and field_dec is not None:
            return self._select(field_ra, field_dec)

        elif fieldgrid is not None:
            return self._iter_grid(fieldgrid)

        else:
            raise ValueError(
                    "Either `field_ra` and `field_dec` must be given or "
                    "`fieldgrid`.")

    #--------------------------------------------------------------------------
    def visualize_selection(
            self, coord_rot_guide, coord_rot_edge, coord_rot_circle):
        """Visualize the guide star selection.

        Parameters
        ----------
        coord_rot_guide : astropy.coordinates.SkyCoord
            Coordinates of the selected guide stars.
        coord_rot_edge : astropy.coordinates.SkyCoord
            Coordinates of the stars in the guide area, but too close to the
            edge.
        coord_rot_circle : astropy.coordinates.SkyCoord
            Coordinates of the stars in the instrument area.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure instance drawn to.
        ax : matplotlib.axes.Axes
            The Axes instance drawn to.
        """

        # plot instrument field:
        offset = self.circle_offset.arcmin
        radius = self.circle_radius.arcmin
        circle = plt.Circle(
                [offset]*2, radius, fill=False, color='k', linestyle='-')
        plt.gca().add_artist(circle)
        plt.plot(offset, offset, marker='+', ms=10, color='k')

        # plot science field:
        field_size = self.field_size.arcmin
        rectangle = plt.Rectangle(
                (-field_size/2, -field_size/2), field_size, field_size,
                fill=False, color='0.5', linestyle='-')
        plt.gca().add_artist(rectangle)
        plt.plot(0, 0, marker='+', ms=10, color='0.5')

        # plot guide area:
        guide_area = self.guide_area.arcmin
        for ((x0, y0), (x1, y1)) in zip(
                    guide_area, np.r_[guide_area[1:], [guide_area[0]]]):
            plt.plot([x0, x1], [y0, y1], color='tab:orange', linestyle='-')

        # plot stars in instrument area:
        ra = coord_rot_circle.ra.arcmin
        ra = np.where(ra>180*60, ra-360*60, ra)
        plt.plot(
                ra, coord_rot_circle.dec.arcmin,
                marker='o', linestyle='None', color='tab:blue')

        # plot stars close to edge:
        ra = coord_rot_edge.ra.arcmin
        ra = np.where(ra>180*60, ra-360*60, ra)
        plt.plot(
                ra, coord_rot_edge.dec.arcmin,
                marker='o', linestyle='None', color='tab:orange', mfc='w')

        # plot guide stars:
        ra = coord_rot_guide.ra.arcmin
        ra = np.where(ra>180*60, ra-360*60, ra)
        plt.plot(
                ra, coord_rot_guide.dec.arcmin,
                marker='o', linestyle='None', color='tab:orange')

        # edit figure:
        plt.gca().set_aspect(1)
        xymin = offset - radius * 1.1
        xymax = offset + radius * 1.1
        plt.xlim(xymin, xymax)
        plt.ylim(xymin, xymax)

        return plt.gcf(), plt.gca()

#==============================================================================
