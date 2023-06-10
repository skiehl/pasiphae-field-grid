# pasiphae-field-grid
Developing the field grid for the Pasiphae survey.

## Directories and files

### Jupyter notebooks

* Develop_FieldGrid.ipynb: Development of the field grid.
* Presentation_FieldGrid.ipynb: Presentation of the first field grid,
    2023-04-24.
* Test_FieldGrid.ipynb: Systematic tests of the grids for gaps and overlap.
* Test_FieldOverlap.ipynb: Cross-checks of the overlapping regions in the grid
    with Gaia sources.

### Scripts

* fieldgrid.py: The main script developed and tested in this repository.
* visualizations.py: Module that provides classes to visualize the field grid
    layouts. This module is copied from the `pasiphae-survey-planner`
    repository and is further developed there.

### Directories

* gaia: Tables of Gaia sources used in the `Test_FieldOverlap.ipynb`.
* gridtests: Pickles and sqlite3 files stored from during the tests performed
    in `Test_FieldGrid.ipynb` and `Test_FieldOverlap.ipynb`.
