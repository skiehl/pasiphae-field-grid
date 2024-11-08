#!/usr/bin/env python3
"""Sky fields for the Pasiphae survey.
"""

from abc import  ABCMeta, abstractmethod
from astropy.coordinates import Angle, SkyCoord
import json
import matplotlib.pyplot as plt
import numpy as np

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

        self.params = None
        self.stars_coord = None
        self.stars_mag = None
        self.fieldgrid = None
        self.guidestars = None
        self.n_guidestars = None
        self.n_fields = None

    #--------------------------------------------------------------------------
    @abstractmethod
    def _locate(self, field_ra, field_dec, return_coord=False):
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

        Notes
        -----
        This method is called by `_select()`.
        """

        pass

    #--------------------------------------------------------------------------
    @abstractmethod
    def _guider_position(self):
        """Calculate the guide camera position for a selection of guide stars.

        Parameters
        ----------
        coord_rot_guide : astropy.coordinates.SkyCoord
            Coordinates of selected guide stars rotated into a reference frame
            that has its origin at the corresponding field center.

        Returns
        -------
        pos_x : numpy.array
            x-position of the guide camera in the units defined by the input
            parameter `scale_xy`.
        pos_y : numpy.array
            y-position of the guide camera in the units defined by the input
            parameter `scale_xy`.

        Notes
        -----
        This method is called by `_select()`.
        """

        pass

    #--------------------------------------------------------------------------
    @abstractmethod
    def _select(self, field_ra, field_dec, return_coord=False, n_max=None):
        """Select guide stars from the candidate star list for a field.

        Parameters
        ----------
        fieldgrid : fieldgrid.FieldGrid, optional
            A field grid. Guide stars will be selected for each field in the
            grid. If not given, then provide `field_ra` and `field_dec`. The
            default is None.
        field_ra : float, optional
            Right ascension of a field center position in radians. Also provide
            `field_dec`. Alternatively, provide `fieldgrid`. The default is
            None.
        field_dec : float, optional
            Declination of a field center position in radians. Also provide
            `field_ra`. Alternatively, provide `fieldgrid`. The default is
            None.
        return_coord : bool, optional
            If True, rotated coordinates of various selected stars are returned
            for plotting with `visualize_selection()`. This option is relevant
            only, when a single field's coordinates are given though `field_ra`
            and `field_dec`. The default is False.
        n_max : int, optional
            Maximum number of guide stars to select for each field. If None,
            all stars in the guider area are saved. The default is None.

        Returns
        -------
        list of dict
            One list entry for each guide star with its right ascension (rad),
            declination (rad), and magnitude and the guide camera position x,
            and y. The guide stars are sorted by increasing magnitude.

        Notes
        -----
        This method is called by `select()` or by `_iter_grid()`.
        This method calls `_locate()` and `_guider_position()`.
        """

        pass

    #--------------------------------------------------------------------------
    def _count_guidestars(self, warn=True):
        """Counts the number of fields and number of guide stars selected for
        each field.

        Parameters
        ----------
        warn : bool, optional
            If True, print out a warning when at least one field does not have
            the required minimum number of guide stars. The default is True.

        Returns
        -------
        None

        Notes
        -----
        This method is called by `select()` and by `_update_guidestars()`.
        """

        self.n_fields = len(self.guidestars)
        self.n_guidestars = np.array([
                len(gs['guidestars']) for gs in self.guidestars])

        if warn and np.any(self.n_guidestars < self.n_min):
            print('WARNING: Not all fields have the required minimum number '
                  f'of guide stars ({self.n_min}) available.\n')

    #--------------------------------------------------------------------------
    def _update_guidestars(self):
        """Update the selection of guide stars for fields that do not have the
        required number of guide stars.

        Returns
        -------
        None

        Notes
        -----
        This method is called by `set_stars()`.
        """

        n_missing = self.n_min - self.n_guidestars
        i_sel = np.nonzero(n_missing > 0)[0]
        n_missing = n_missing[i_sel]
        n = i_sel.shape[0]
        print(f'\n{n} field do not have enough guide stars. Search for more..')

        # iterate though fields with insufficient number of guide stars:
        for i, (j, n_max) in enumerate(zip(i_sel, n_missing)):
            print(f'\rField {i} of {n} ({i/n*100:.1f}%)..', end='')

            # select guide stars:
            field_ra = self.guidestars[j]['field_center_ra']
            field_dec = self.guidestars[j]['field_center_dec']
            guidestars = self._select(field_ra, field_dec, n_max=n_max)

            # append guide stars:
            self.guidestars[j]['guidestars'] += guidestars

        print('\r  done.                             \n')
        self._count_guidestars()
        self.check_results()

    #--------------------------------------------------------------------------
    def _iter_grid(self, fieldgrid):
        """

        Parameters
        ----------
        fieldgrid : fieldgrid.FieldGrid, optional
            A field grid. Guide stars will be selected for each field in the
            grid.

        Returns
        -------
        ist of dict
            One list entry for each field. Each dict contains the field center
            coordinates and a list of selected guide stars. This list contains
            a dict for each guide star with its right ascension (rad),
            declination (rad), and magnitude and the guide camera position x,
            and y.

        Notes
        -----
        This method is called by `select()`.
        This method calls `_select()`.
        """

        field_ras, field_decs = fieldgrid.get_center_coords()
        n = len(fieldgrid)
        print('Iterate through field grid..')
        guidestars = []

        for i, (field_ra, field_dec) in enumerate(zip(field_ras, field_decs)):
            print(f'\rField {i} of {n} ({i/n*100:.1f}%)..', end='')

            guidestars_for_field = self._select(field_ra, field_dec)
            guidestars.append({
                    'field_center_ra': field_ra, 'field_center_dec': field_dec,
                    'guidestars': guidestars_for_field})

        print('\r  done.                             \n')

        return guidestars

    #--------------------------------------------------------------------------
    def _set_stars(self, ra, dec, mag):
        """Store the stars' coordinates and magnitudes, from which guide stars
        will be selected.

        Parameters
        ----------
        ra : array-like
            Right ascensions of the stars in radians.
        dec : array-like
            Declinations of the stars in radians.
        mag : array-like
            Magnitudes of the stars.

        Notes
        -----
        This method is called by `set_stars()`.
        """

        self.stars_coord = SkyCoord(ra, dec, unit='rad')
        self.stars_mag = np.asarray(mag)
        self.stars_mag_min = self.stars_mag.min()
        self.stars_mag_max = self.stars_mag.max()

    #--------------------------------------------------------------------------
    def set_stars(self, ra, dec, mag):
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

        Notes
        -----
        This method calls `_set_stars()` and, if the list of stars is updated,
        `_update_guidestars()`.
        """

        # set stars for the first time:
        if self.stars_mag is None:
            self._set_stars(ra, dec, mag)
            print(f'{len(ra)} candidate stars added.')
            print(f'Magnitude range: {self.stars_mag_min:.1f} - '
                  f'{self.stars_mag_max:.1f}')

        # update stars:
        else:
            mag = np.asarray(mag)

            # check that new stars have higher magnitudes than previous stars:
            if mag.min() < self.stars_mag_max:
                raise ValueError(
                        "New stars must have higher magnitudes than "
                        f"previously set stars: > {self.stars_mag_max}")

            # update variables:
            print('Overwriting previous stars..')
            self._set_stars(ra, dec, mag)
            print(f'{len(ra)} candidate stars added.')
            print(f'Magnitude range: {self.stars_mag_min:.1f} - '
                  f'{self.stars_mag_max:.1f}')
            self._update_guidestars()

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
    def select(
            self, fieldgrid=None, field_ra=None, field_dec=None,
            return_coord=False, verbose=1):
        """Select guide stars from the candidate star list for either each
        field in a field grid or for a specific field's coordinates.

        Parameters
        ----------
        fieldgrid : fieldgrid.FieldGrid, optional
            A field grid. Guide stars will be selected for each field in the
            grid. If not given, then provide `field_ra` and `field_dec`. The
            default is None.
        field_ra : float, optional
            Right ascension of a field center position in radians. Also provide
            `field_dec`. Alternatively, provide `fieldgrid`. The default is
            None.
        field_dec : float, optional
            Declination of a field center position in radians. Also provide
            `field_ra`. Alternatively, provide `fieldgrid`. The default is
            None.
        return_coord : bool, optional
            If True, rotated coordinates of various selected stars are returned
            for plotting with `visualize_selection()`. This option is relevant
            only, when a single field's coordinates are given though `field_ra`
            and `field_dec`. The default is False.
        verbose : int, optional
            Controls the level of detail of information printed. The default is
            1.

        Raises
        ------
        ValueError
            Raised, if neither `field_ra` and `field_dec` or `fieldgrid` are
            given.

        Returns
        -------
        list of dict
            One list entry for each field. Each dict contains the field center
            coordinates and a list of selected guide stars. This list contains
            a dict for each guide star with its right ascension (rad),
            declination (rad), and magnitude and the guide camera position x,
            and y.
        or
        list of dict, astropy.coordinates.SkyCoord,
        astropy.coordinates.SkyCoord, astropy.coordinates.SkyCoord
            Returned if `return_coord=True` and a single field's coordinates
            are given though `field_ra` and `field_dec`. The list of dict has
            the same structure as above. Additionally, the coordinates of the
            following selected stars are provided: the selected guide stars,
            guide star candidates in the guider area but too close to the edge,
            guide star candidates in the circular field of view. All
            coordinates are rotated to a system where the field center is at
            (0, 0), for direct use in the `visualize_selection()` method.

        Notes
        -----
        This method calls `_select()` for a single field or `_iter_grid()`,
        `_count_guidestars()`, and `check_results()` for a field grid.
        """

        # select guide stars for single field:
        if field_ra is not None and field_dec is not None:
            if return_coord:
                guidestars, coord_rot_guide, coord_rot_edge, coord_rot_circle \
                        = self._select(field_ra, field_dec, return_coord=True)

                return (guidestars, coord_rot_guide, coord_rot_edge,
                        coord_rot_circle)

            else:
                guidestars = self._select(field_ra, field_dec)

        # select guide stars for entire field grid:
        elif fieldgrid is not None:
            guidestars = self._iter_grid(fieldgrid)
            self.guidestars = guidestars
            self._count_guidestars()
            self.check_results(verbose=verbose)

        else:
            raise ValueError(
                    "Either `field_ra` and `field_dec` must be given or "
                    "`fieldgrid`.")

        return guidestars

    #--------------------------------------------------------------------------
    def check_results(self, verbose=1):
        """Print out various statistics about the selected guide stars.

        Parameters
        ----------
        verbose : int, optional
            Controls the level of detail of information printed. The default is
            1.

        Returns
        -------
        None

        Notes
        -----
        This method is called by `select()` and by `_update_guidestars()`.
        """

        if verbose > 0:
            n_fields = self.n_fields
            n_total = self.n_guidestars.sum()
            n_zero = (self.n_guidestars == 0).sum()
            n_median = np.median(self.n_guidestars)
            n_mean = np.mean(self.n_guidestars)
            n_max = np.max(self.n_guidestars)

            print('Results:')
            print('--------------------------------------------')
            print(f'Guide stars selected:         {n_total:6.0f}')
            print(f'Fields without guide stars:   {n_zero:6d} '
                  f'({n_zero/n_fields*100:.1f}%)')
            print('--------------------------------------------')
            print(f'Median number of field stars: {n_median:6.0f}')
            print(f'Mean number of field stars:   {n_mean:6.0f}')
            print(f'Max number of field stars:    {n_max:6.0f}')

        if verbose > 1:
            print('--------------------------------------------')
            print('No. of guide stars: No. of fields')

            for value, count in zip(
                    *np.unique(self.n_guidestars, return_counts=True)):
                print(f'{value:2.0f}: {count:3d}')

            print('--------------------------------------------')

#==============================================================================

class GuideStarWalopS(GuideStarSelector):
    """A class to select guide stars for WALOP-South targets."""

    #--------------------------------------------------------------------------
    def _locate(self, field_ra, field_dec, return_coord=False):
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

        Notes
        -----
        This method is called by `_select()`.
        """

        # select closest stars
        radius = self.circle_radius - self.limit
        instrument_center = SkyCoord(
                field_ra + self.circle_offset.rad,
                field_dec + self.circle_offset.rad,
                unit='rad')
        sel_circle = self.stars_coord.separation(instrument_center) < radius
        i_circle = np.nonzero(sel_circle)[0]
        candidates_coord = self.stars_coord[sel_circle]

        # rotate coordinate frame:
        ra_rot, dec_rot = rotate_frame(
                candidates_coord.ra.rad, candidates_coord.dec.rad,
                instrument_center, tilt=self.instr_rot.rad)
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
    def _guider_position(self, coord_rot_guide):
        """Calculate the guide camera position for a selection of guide stars.

        Parameters
        ----------
        coord_rot_guide : astropy.coordinates.SkyCoord
            Coordinates of selected guide stars rotated into a reference frame
            that has its origin at the corresponding field center.

        Returns
        -------
        pos_x : numpy.array
            x-position of the guide camera in the units defined by the input
            parameter `scale_xy`.
        pos_y : numpy.array
            y-position of the guide camera in the units defined by the input
            parameter `scale_xy`.

        Notes
        -----
        This method is called by `_select()`.
        """

        pos_x = (coord_rot_guide.ra - self.home_pos[0]).rad * -self.scale_xy
        pos_y = (coord_rot_guide.dec - self.home_pos[1]).rad * -self.scale_xy

        return pos_x, pos_y

    #--------------------------------------------------------------------------
    def _select(self, field_ra, field_dec, return_coord=False, n_max=None):
        """Select guide stars from the candidate star list for a field.

        Parameters
        ----------
        fieldgrid : fieldgrid.FieldGrid, optional
            A field grid. Guide stars will be selected for each field in the
            grid. If not given, then provide `field_ra` and `field_dec`. The
            default is None.
        field_ra : float, optional
            Right ascension of a field center position in radians. Also provide
            `field_dec`. Alternatively, provide `fieldgrid`. The default is
            None.
        field_dec : float, optional
            Declination of a field center position in radians. Also provide
            `field_ra`. Alternatively, provide `fieldgrid`. The default is
            None.
        return_coord : bool, optional
            If True, rotated coordinates of various selected stars are returned
            for plotting with `visualize_selection()`. This option is relevant
            only, when a single field's coordinates are given though `field_ra`
            and `field_dec`. The default is False.
        n_max : int, optional
            Maximum number of guide stars to select for each field. If None,
            all stars in the guider area are saved. The default is None.

        Returns
        -------
        list of dict
            One list entry for each guide star with its right ascension (rad),
            declination (rad), and magnitude and the guide camera position x,
            and y. The guide stars are sorted by increasing magnitude.

        Notes
        -----
        This method is called by `select()` or by `_iter_grid()`.
        This method calls `_locate()` and `_guider_position()`.
        """

        # locate guide stars in the guider area:
        i_guide, coord_rot_guide, coord_rot_edge, coord_rot_circle \
                = self._locate(field_ra, field_dec, return_coord=True)

        # determine coordinates and magnitudes:
        guidestars_coord = self.stars_coord[i_guide]
        guidestars_mag = self.stars_mag[i_guide]

        # if maximum number of guide stars exceed, select brightest ones:
        if n_max is None:
            n_max = self.n_max

        if n_max and guidestars_mag.shape[0] > n_max:
            i_sort = np.argsort(guidestars_mag)[:n_max]
            guidestars_coord = guidestars_coord[i_sort]
            guidestars_mag = guidestars_mag[i_sort]

        # otherwise, sort by brightness:
        else:
            i_sort = np.argsort(guidestars_mag)
            guidestars_coord = guidestars_coord[i_sort]
            guidestars_mag = guidestars_mag[i_sort]

        # determine guider camera positions:
        pos_x, pos_y = self._guider_position(coord_rot_guide)

        # prepare list of guide stars:
        guidestars = []

        for coord, mag, x, y in zip(
                guidestars_coord, guidestars_mag, pos_x, pos_y):
            guidestars.append({
                    'guidestar_ra': coord.ra.rad,
                    'guidestar_dec': coord.dec.rad,
                    'guidestar_mag': mag,
                    'cam_pos_x': x,
                    'cam_pos_y': y})

        if return_coord:
            return (guidestars, coord_rot_guide, coord_rot_edge,
                    coord_rot_circle)
        else:
            return guidestars

    #--------------------------------------------------------------------------
    def set_params(
            self, circle_radius, circle_offset, field_size, guide_area,
            home_pos, instr_rot=0, limit=0, scale=1, scale_xy=1, n_min=1,
            n_max=0):
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
        home_pos : array-like
            The home position of the guide camera. Can be given in any unit.
            Multiplied with the `scale` factor that converts unit to radians.
        instr_rot : float, optional
            Instrument rotation in radians, counted clockwise. Use if the
            instrument is mounted such that the guide area is not facing North-
            East. The default is 0.
        limit : float, optional
            Only stars located off the edge of the guide area by at least this
            limit are selected as guide stars. Can be given in any unit.
            Multiplied with the `scale` factor that converts unit to radians.
            The default is 0.
        scale : float, optional
            Scale factor that converts the above values to radians. The default
            is 1.
        scale_xy : float, optional
            Scale factor that converts the guide camera position from radians
            to the designated unit. The default is 1.
        n_min : int, optional
            The minimum number of guide stars intended for each field. A
            warning is printed if some fields end up having fewer guide stars.
            If additional stars are added through `set_stars()`, fields that
            do not reach the minimum number are updated. The default is 1.
        n_max : int, optional
            The maximum number of guide stars selected for a field. If more are
            available, the brightest ones are selected. If 0, all available
            stars are selected. The default is 0.

        Raises
        ------
        ValueError
            Raised, if `scale` is negative or zero.
            Raised, if `scale_xy` is negative or zero.
            Raised, if `limit` is negative.
            Raised, if `guide_area` is not 2-dimensional.
            Raised, if `home_pos` does not consist of two numbers.
            Raised, if `instr_rot` is not between 0 and 2*pi.
            Raised, if `n_min` is not int or is smaller than 1.
            Raised, if `n_max` is not int or is smaller than 0.

        Returns
        -------
        None
        """

        guide_area = np.asarray(guide_area)
        home_pos = np.asarray(home_pos)

        # check inputs:
        if scale <= 0:
            raise ValueError("`scale` must be > 0.")
        if scale_xy <= 0:
            raise ValueError("`scale_xy` must be > 0.")
        if limit < 0:
            raise ValueError("`limit` must be >= 0.")
        if guide_area.ndim != 2:
            raise ValueError("`guide_area` must be 2-dimensional.")
        if home_pos.shape[0] != 2 or home_pos.ndim != 1:
            raise ValueError("`home_pos` must consist of two numbers.")
        if instr_rot < 0 or instr_rot >= 2 * np.pi:
            raise ValueError("`instr_rot` must be >=0 and <2*pi.")
        if not isinstance(n_min, int) or n_min < 1:
            raise ValueError("`n_min` must be int >= 1.")
        if not isinstance(n_min, int) or n_max < 0:
            raise ValueError("`n_max` must be int >= 0.")

        # store parameters:
        self.params = {
                'circle_radius': circle_radius,
                'circle_offset': circle_offset,
                'field_size': field_size,
                'guide_area': guide_area.tolist(),
                'home_pos': home_pos.tolist(),
                'instr_rot': instr_rot,
                'limit': limit,
                'scale': scale,
                'scale_xy': scale_xy,
                'n_min': n_min,
                'n_max': n_max}
        self.circle_radius = Angle(circle_radius*scale, unit='rad')
        self.circle_offset = Angle(circle_offset*scale, unit='rad')
        self.field_size = Angle(field_size*scale, unit='rad')
        self.guide_area = \
                Angle(guide_area*scale, unit='rad') - self.circle_offset
        self.home_pos = Angle(home_pos*scale, unit='rad')
        self.instr_rot = Angle(instr_rot, unit='rad')
        self.limit = Angle(limit*scale, unit='rad')
        self.scale_xy = scale_xy
        self.n_min = n_min
        self.n_max = n_max

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
                [0, 0], radius, fill=False, color='k', linestyle='-')
        plt.gca().add_artist(circle)
        plt.plot(0, 0, marker='+', ms=10, color='k')

        # plot science field:
        field_size = self.field_size.arcmin
        rectangle = plt.Rectangle(
                (-field_size/2-offset, -field_size/2-offset), field_size,
                field_size, fill=False, color='0.5', linestyle='-')
        plt.gca().add_artist(rectangle)
        plt.plot(-offset, -offset, marker='+', ms=10, color='0.5')

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
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_aspect(1)
        xymin = offset - radius * 1.1
        xymax = offset + radius * 1.1
        plt.xlim(xymin, xymax)
        plt.ylim(xymin, xymax)

        return fig, ax

#==============================================================================
