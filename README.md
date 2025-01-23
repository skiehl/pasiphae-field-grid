# pasiphae-field-grid

## About
The [Pasiphae project](http://pasiphae.science/) aims to map, with
unprecedented accuracy, the polarization of millions of stars at areas of the
sky away from the Galactic plane, in both the Northern and the Southern
hemispheres. New instruments, calibration and analysis methods, and dedicated
software are currently under development.

The pasiphae-field-grid package provides classes to separate the sky into
fields. Different layouts optionally allow the user to set declination limits
and to exclude the Galactic plane up to a specified Galactic latitude limit.

## Modules

* `fieldgrid.py`: Classes to separate the sky into different field grids.
    * `FieldGrid` class: Parent class that provides methods shared by the
      following two field grid classes and that defines some abstract methods.
    * `FieldGridIsoLat` class: Devide the sky into fields that fill up rings
      of constant latitudes. The number of fields per ring is automatically
      determined to ensure the least amount of fields without having any
      observational gaps on the sky. This is the basic field grid layout used
      by the Pasiphae survey.
    * `FieldGridGrtCirc` class: Devide the sky into fields along great cicles.
      The number of great circles and the number of fields along each circle is
      automatically chosen based on the field of view. This is an inefficient
      approach to split the sky into fields, because it results in strong
      overlap closer to the poles.
    * `FieldGridTester` class: Provides methods to test how strongly
      neighboring fields overlap and whether gaps exist in the field grid.
* `utilities.py`: Provides various utility functions.
* `visualizations.py`: Class to visualize the field grid layouts.
    * `FieldGridVisualizer` class: Visualize the field grid in orthographic or
      Mollweide projection.

## Notebooks

* `Develop_Fieldgrid.ipynb`: Development of the field grid layouts,
  visualizations, test methods, and package implementation.
* `Develop_Guidestar_N.ipynb`: Development of the WALOP-N guide star selection
  code.
* `Test_FieldGrid.ipynb`: Tests of different field grid layouts, including
  checks for overlap of neighboring fields and gaps in the grid.
* `Test_FieldGridSetups.ipynb`: Tests by how much the number of fields in the
  grid is increased by different choices for the overlap between neighboring
  fields and by requiring full coverage at the Galactic plane limits.
* `Test_FieldOverlap.ipynb`: Testing how many Gaia stars are contained in the
  overlapping regions. This notebook is based on an outdated version of
  fieldgrid.py.
* `Presentation_FieldGrid.ipynb`: A shorter version of `Test_FieldGrid.ipynb`
  focused on presenting the main results.

## Auxiliary files
* `*.json`: Contain parameters for setting up the field grids for the
  Pasiphae survey of for testing the code with a coarser grid.
