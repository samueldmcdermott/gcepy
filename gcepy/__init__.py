"""
code for sampling for testing models of the Galactic center excess (GCE)
can use highdim_model and lowdim_model separately, or use lnlike() directly
data for lowdim_model is supplied and will run out of the box
data for highdim_model must be requested and patched in manually or via appropriate edit
"""

__version__ = "0.2"

import gcepy.lowdim_model as _lm
del(lowdim_model)

try:
    import gcepy.highdim_model as _hm
    del(highdim_model)
    no_hm = False
except FileNotFoundError:
    no_hm = True
    print("high dimensional model not available")

def lnlike(model, *args, **kwargs):
    if 'l' in model:
        return _lm.jjlnprob(*args, **kwargs)
    elif 'h' in model:
        if no_hm:
            print("high dimensional model not available")
            pass
        else:
            return _hm.jjlnprob(*args, **kwargs)
    else:
        print("first argument must include 'l' (for low-dimensional) or 'h' (for high-dimensional), please")