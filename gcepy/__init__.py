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
    """
    
    Parameters
    ----------
    model: str
        must contain "l" to specify the low-dimensional model or "h" to specify the high-dimensional model (if you
        have that model available). Optionally, can include "s" to specify to the "smooth" prior, which is necessary
        for mchmc (the uniform priors get smoothed by sigmoids)
    args:
        there is one argument, which is the parameter vector for the likelihood
    kwargs:
        bin_no (specifying the energy bin) and ex_num (specifying the 'example number' or model (differs for low- vs
        high-dimensional likelihoods))

    Returns
        a jax float
    -------

    """
    if 'l' in model:
        if "s" in model:
            return _lm.jjlnprob_smooth(*args, **kwargs)
        else:
            return _lm.jjlnprob_hard(*args, **kwargs)
    elif 'h' in model:
        if no_hm:
            print("high dimensional model not available")
            pass
        else:
            if "s" in model:
                return _hm.jjlnprob_smooth(*args, **kwargs)
            else:
                return _hm.jjlnprob_hard(*args, **kwargs)
    else:
        print("first argument must include 'l' (for low-dimensional) or 'h' (for high-dimensional), please."
              "Also include 's' if you want to have smooth priors (necessary for mchmc).")