"""
code for sampling for testing models of the Galactic center excess (GCE)
can use highdim_model and lowdim_model separately, or use lnlike() directly
data for lowdim_model is supplied and will run out of the box
data for highdim_model must be requested and patched in manually or via appropriate edit
"""

__version__ = "0.2"

from gcepy.lowdim_model import jjlnprob as _lmlnprob
from gcepy.highdim_model import jjlnprob as _hmlnprob

def lnlike(model, *args, **kwargs):
    if model == 'low':
        return _lmlnprob(*args, **kwargs)
    elif model == 'high':
        return _hmlnprob(*args, **kwargs)
    else:
        print("choose 'low' or 'high' as the first argument, please")
