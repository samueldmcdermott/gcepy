from jax import random as jr
from jax import numpy as jnp

import gcepy.models.lowdim_model as _lm

try:
    import gcepy.models.highdim_model as _hm
    no_hm = False
except FileNotFoundError:
    no_hm = True
    print("high dimensional model not available")

fermi_dat = jnp.array([_lm.fermi_front_20x20[i] for i in range(_lm.num_ebins)])
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
    -------
        a jax float
    """
    if 'l' in model:
        if "s" in model:
            return _lm.jjlnprob_smooth(*args, **kwargs)
        else:
            return _lm.jjlnprob_hard(*args, **kwargs)
    elif 'h' in model:
        if no_hm:
            print("high dimensional model not available")
            return None
        else:
            if "s" in model:
                return _hm.jjlnprob_smooth(*args, **kwargs)
            else:
                return _hm.jjlnprob_hard(*args, **kwargs)
    else:
        print("first argument must include 'l' (for low-dimensional) or 'h' (for high-dimensional), please."
              "Also include 's' if you want to have smooth priors (necessary for mchmc).")

def expected_mock_sky(model, *args, **kwargs):
    """
    Parameters
    ----------
    model: str
        must contain "l" to specify the low-dimensional model or "h" to specify the high-dimensional model (if you
        have that model available)
    args:
        there are two arguments: `theta` which specify the parameter values and `bin_no` which specifies the energy bin
    kwargs:
        ex_num (specifying the 'example number', which determines the model (differs for low- vs high-dimensional likelihoods))

    Returns
    -------
        the value in expectation of the gamma-ray sky for a given set of parameters
    """
    if (('l' not in model) and ('h' not in model)):
        print("first argument must include 'l' (for low-dimensional) or 'h' (for high-dimensional), please."
              "Also include 's' if you want to have smooth priors (necessary for mchmc).")
        return None
    elif 'h' in model:
        if no_hm:
            print("high dimensional model not available")
            return None
        else:
            sky = _hm.jmodel_masked(*args, **kwargs)
    else:
        sky = _lm.jmodel_masked(*args, **kwargs)
    return sky.reshape(int(len(sky)**0.5), -1) if len(sky.shape) == 1 else sky

def single_poisson_mock_sky(model, jrkey, *args, **kwargs):
    """
    Parameters
    ----------
    model: str
        must contain "l" to specify the low-dimensional model or "h" to specify the high-dimensional model (if you
        have that model available)
    jrkey:
        this is a jax PRNGKey array-type argument which gets fed directly into the jax generator
    args:
        there are two arguments: `theta` which specify the parameter values and `bin_no` which specifies the energy bin,
    kwargs:
        ex_num (specifying the 'example number', which determines the model (differs for low- vs high-dimensional likelihoods))

    Returns
    -------
        a Poisson sample of the gamma-ray sky for a given set of parameters
    """
    sky = expected_mock_sky(model, *args, **kwargs)
    return jr.poisson(jrkey, sky)

def multi_poisson_mock_sky(model, jrkey, n_mocks, theta0, bin_no, rel_unc = 0.1, return_thetas = False, **kwargs):
    """
    Parameters
    ----------
    model: str
        must contain "l" to specify the low-dimensional model or "h" to specify the high-dimensional model (if you
        have that model available)
    jrkey:
        this is a jax PRNGKey array-type argument which gets fed directly into the jax generator
    n_mocks:
        number of mock skies to generate
    theta0:
        central value of the parameters to simulate around
    bin_no:
        energy bin
    rel_unc: 0.1, float
        relative uncertainty of the thetas
    kwargs:
        ex_num (specifying the 'example number', which determines the model (differs for low- vs high-dimensional likelihoods))

    Returns
    -------
        a Poisson sample of the gamma-ray sky for a given set of parameters
    """
    theta_scale = jr.normal(jrkey, (n_mocks, len(theta0)))*rel_unc
    thetas = theta0[None,:] + theta_scale
    skies = jnp.array([expected_mock_sky(model, theta = t, bin_no=bin_no, **kwargs) for t in thetas])
    jrkey, jrkey_ = jr.split(jrkey)
    if not return_thetas:
        return jr.poisson(jrkey, skies)
    else:
        return thetas, jr.poisson(jrkey, skies)