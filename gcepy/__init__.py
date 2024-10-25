"""
code for sampling for testing models of the Galactic center excess (GCE)
can use highdim_model and lowdim_model separately, or use lnlike() directly
data for lowdim_model is supplied and will run out of the box
data for highdim_model must be requested and patched in manually or via appropriate edit
"""

__version__ = "0.2"

from gcepy.gcepy import fermi_dat, lnlike, multi_poisson_mock_sky, single_poisson_mock_sky, expected_mock_sky