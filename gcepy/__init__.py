"""
code for sampling for testing models of the Galactic center excess (GCE)
can use highdim_model and lowdim_model separately, or use lnlike() directly
data for lowdim_model is supplied and will run out of the box
data for highdim_model must be requested and patched in manually or via appropriate edit
"""

__version__ = "0.2"

import gcepy.lowdim_model as _lm
del(lowdim_model)
import gcepy.highdim_model as _hm
del(highdim_model)

def lnlike(model, *args, **kwargs):
    if 'l' in model:
        return _lm.jjlnprob(*args, **kwargs)
    elif 'h' in model:
        return _hm.jjlnprob(*args, **kwargs)
    else:
        print("must include 'l' (for low-dimensional) or 'h' (for high-dimensional) when you specify the first "
              "argument, please")