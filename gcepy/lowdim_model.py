"""
defines a jax-based likelihood for template analysis of the GCE
the data is loaded at first and is _not_ included in this repository -- you must request permission from us as well as the first author of the relevant paper
the outputs of most interest are:
    jjlnlike, which takes as arguments a vector of normalizations, a bin number, and a number describing the excess (0=none, 1=DM, 2=boxy bulge, 3=x-shaped bulge, 4=boxy bulge plus nuclear bulge, 5=boxy bulge and x-shaped bulge varying independently)
    and
    jjlnprob, which is the same as jjlnlike but enforces predetermined log priors
"""

from jax import jit
import jax.numpy as jnp
import jax.scipy.special as jsc
import os
import yaml

gcepydir = os.path.dirname(os.path.abspath(__file__))

EPS = jnp.finfo(jnp.float32).eps #smallest machine-precision value
EMAX = jnp.finfo(jnp.float32).max #largest machine-precision value

num_ebins = 14 #we use 14 energy bins

# gcepydir = os.getcwd() 
utils_dir = gcepydir + '/inputs/utils/' #location of your data, mask, etc
templates_dir = gcepydir + '/inputs/templates_lowdim/' #location of your model templates
excesses_dir = gcepydir + '/inputs/excesses/' #location of your excess templates

suffix = '_front_only_14_Ebin_20x20window_normal.npy' #this is convenient in case you have any other labels attached to the models

fermi_front_20x20 = jnp.load(utils_dir + 'fermi_w009_to_w670' + suffix).reshape(num_ebins, -1) #the data we used
mask_20x20 = jnp.load(utils_dir + 'mask_4FGL-DR2_14_Ebin_20x20window_normal.npy').reshape(num_ebins, -1) #this is the point source _and_ disk mask

#the next two lines give the bubbles and isotropic templates
bubble_20x20 = jnp.load(utils_dir + 'bubble' + suffix).reshape(num_ebins, -1)
isotropic_20x20 = jnp.load(utils_dir + 'isotropic' + suffix).reshape(num_ebins, -1)
isotropic_error, bubble_error = jnp.load(utils_dir +'external_errors.npy') #the denominators on the isotropic and Bubble normalization terms in the "external chi^2"


#the next six lines give the three astrophysical templates for the best fit models with and without an excess from 2112.09706
bremss = jnp.load(templates_dir+"bremss_model_8t_front_only_14_Ebin_20x20window_normal.npy").reshape(14, -1)
pi0 = jnp.load(templates_dir+"pion0_model_8t_front_only_14_Ebin_20x20window_normal.npy").reshape(14, -1)
ics = jnp.load(templates_dir+"ics_model_8t_front_only_14_Ebin_20x20window_normal.npy").reshape(14, -1)
bremss_nodm = jnp.load(templates_dir+"bremss_model_7p_front_only_14_Ebin_20x20window_normal.npy").reshape(14, -1)
pi0_nodm = jnp.load(templates_dir+"pion0_model_7p_front_only_14_Ebin_20x20window_normal.npy").reshape(14, -1)
ics_nodm = jnp.load(templates_dir+"ics_model_7p_front_only_14_Ebin_20x20window_normal.npy").reshape(14, -1)



dm_20x20 = jnp.load(excesses_dir + 'dm' + suffix).reshape(num_ebins, -1) #the emission expected from annihilation of dark matter; we assume it follows a gNFW morphology with gamma=1.2 and has the energy spectrum of a 30 GeV particle annihilating to b \bar{b} (though we fit every energy bin independently)
bb_20x20 = jnp.load(excesses_dir + 'bb' + suffix).reshape(num_ebins, -1) #the profile of the boxy bulge; this has a power-law energy distribution across bins
x_20x20 = jnp.load(excesses_dir + 'x' + suffix).reshape(num_ebins, -1) #the profile of the x-shaped bulge; this has a power-law energy distribution across bins
bbp_20x20 = jnp.load(excesses_dir + 'bbp' + suffix).reshape(num_ebins, -1) #the "boxy bulge plus" = the profile of the boxy bulge augmented with the nuclear stellar bulge and nuclear disk; this has a power-law energy distribution across bins




#jax requires arrays to be of fixed size, so we can't for example write a function that starts with the astrophysical background and adds additional templates later
#instead we have to write out each possibility on its own

#here is the model prediction in a single energy bin given a list of theta = log10(normalizations)
def jmodel_masked(theta, bin_no, ex_num=1):
    """
    function that returns a predicted (masked) model of the gamma-ray sky
    
    Parameters
    ----------
    theta: vector
        log10 of the normalizations of the different emission components
        the first 4 are the best-fit templates of https://arxiv.org/abs/2112.09706 (these are different if there is or is not an excess, which we account for) -- these are identical to the templates at https://zenodo.org/record/6423495 but already in .npy form
    
    bin_no: int
        the energy bin number
    
    ex_num: int
        a way of selecting the excess to add to the astrophysical rings
        0 corresponds to no excess
        1 corresponds to DM
        2 is the boxy bulge
        3 is the x-shaped bulge
        4 is the 'stellar bulge', boxy plus nuclear with a certain ratio constrained in a certain way, aka dm_64
        5 allows both the DM and the boxy bulge to be free independently
    
    Returns
    -------
    jnp.array
        a 160,000-entry vector (including the appropriate mask) that can be compared to data
    """
    if ex_num==1:#DM
        out = jnp.einsum('i,ij->j', 10**theta, jnp.array([bremss[bin_no] + pi0[bin_no], ics[bin_no], bubble_20x20[bin_no], isotropic_20x20[bin_no], dm_20x20[bin_no]]))
    elif ex_num==2:#boxy bulge
        out = jnp.einsum('i,ij->j', 10**theta, jnp.array([bremss[bin_no] + pi0[bin_no], ics[bin_no], bubble_20x20[bin_no], isotropic_20x20[bin_no], bb_20x20[bin_no]]))
    elif ex_num==3:#x-shaped bulge
        out = jnp.einsum('i,ij->j', 10**theta, jnp.array([bremss[bin_no] + pi0[bin_no], ics[bin_no], bubble_20x20[bin_no], isotropic_20x20[bin_no], x_20x20[bin_no]]))
    elif ex_num==4:#boxy bulge plus
        out = jnp.einsum('i,ij->j', 10**theta, jnp.array([bremss[bin_no] + pi0[bin_no], ics[bin_no], bubble_20x20[bin_no], isotropic_20x20[bin_no], bbp_20x20[bin_no]]))
    elif ex_num==5:#DM & boxy bulge plus, independently
        out = jnp.einsum('i,ij->j', 10**theta, jnp.array([bremss[bin_no] + pi0[bin_no], ics[bin_no], bubble_20x20[bin_no], isotropic_20x20[bin_no], dm_20x20[bin_no], bbp_20x20[bin_no]]))
    else:#no excess
        out = jnp.einsum('i,ij->j', 10**theta, jnp.array([bremss_nodm[bin_no] + pi0_nodm[bin_no], ics_nodm[bin_no], bubble_20x20[bin_no], isotropic_20x20[bin_no]]))
    return jnp.where(jnp.asarray(mask_20x20)[bin_no]==0, 0, out)
jjmodel_masked = jit(jmodel_masked, static_argnums=(1,2))


#masked data
def jdata_masked(bin_no):
    """
    function that returns the masked data of the gamma-ray sky
    
    Parameters
    ----------
    bin_no: int
        the energy bin number
    
    Returns
    -------
    jnp.array
        a 160,000-entry vector (including the appropriate mask) to which a model can be compared
    """
    return jnp.where(jnp.asarray(mask_20x20)[bin_no]==0, 0, jnp.asarray(fermi_front_20x20)[bin_no])
jjdata_masked = jit(jdata_masked, static_argnums=(0,))


#the log of the Poisson likelihood plus the "external chi^2" terms
def jlnlike(theta, bin_no=0, ex_num=1):
    """
    function that returns a log-likelihood that the data is described by a model of the gamma-ray sky
    
    Parameters
    ----------
    theta: vector
        log10 of the normalizations of the different emission components
        the first 18 are the astrophysical rings of Pohl et al.
    
    bin_no: int
        the energy bin number
    
    ex_num: int
        a way of selecting the excess to add to the astrophysical rings
        0 corresponds to no excess
        1 corresponds to DM
        2 is the boxy bulge
        3 is the x-shaped bulge
        4 is the 'stellar bulge', boxy plus nuclear with a certain ratio constrained in a certain way, aka dm_64
        5 allows both the boxy bulge and the x-shaped bulge to be free independently
    
    Returns
    -------
    float
        the log-likelihood _from the constrained space_ that the data is described by a model of the gamma-ray sky
    """
    bubble_norm, isotropic_norm = 10**theta[2], 10**theta[3]
    
    e, d = jjmodel_masked(theta, bin_no, ex_num), jjdata_masked(bin_no)
    mylike = 2*(jnp.sum(e+jsc.gammaln(d+1) - jsc.xlogy(d, e+EPS)))#gammaln takes the log of the gamma function,
    # which itself is the factorial shifted by one; we use jsc.xlogy for the second log
    # because we are including masked pixels in the sum, in which the expectation is not necessarily zero, but which
    # are multiplied by exactly zero since the data is also masked, and rather than omit them from the sum (which jax
    # dislikes) we use a special function which is zero when d is zero
    chi2extBub = ((bubble_norm-1.0)/bubble_error[bin_no])**2
    chi2extIso = ((isotropic_norm-1.0)/isotropic_error[bin_no])**2
    
    return -.5*(mylike + chi2extBub + chi2extIso)
jjlnlike = jit(jlnlike, static_argnums=(1,2))


#the priors
with open(os.path.join(gcepydir, "inputs", "priors", "lowdim_priors.yaml"), "r") as f:
    _ = yaml.safe_load(f)
    pmin, pmax = jnp.asarray(_['low']), jnp.asarray(_['high'])


# HARD PRIOR -- numpyro HMC prefers this
# returns the negative of the machine-precision-large number if we find ourselves outside of the prior range
# this is only necessary for samplers that work in the unconstrained space, which is true of numpyro
def jlnprior_hard(theta):
    """
    function that returns a big negative number if you violate the priors

    Parameters
    ----------
    theta: vector
        log10 of the normalizations of the different emission components
        the first 18 are the astrophysical rings of Pohl et al.

    Returns
    -------
    float
        zero if you respect the priors, or a big negative number if you violate the priors
    """
    return -(1 - jnp.prod(theta > pmin[:len(theta)]) * jnp.prod(theta < pmax[:len(theta)])) * EMAX
jjlnprior_hard = jit(jlnprior_hard)


# SMOOTH PRIOR -- mchmc prefers this
# adds the logs of two sigmoids centered on the low and high edges of the prior range
# this is only necessary for samplers that work in the unconstrained space, which is true of (MC)HMC
def jlnprior_smooth(theta):
    """
    function that returns a big negative number if you violate the priors

    Parameters
    ----------
    theta: vector
        log10 of the normalizations of the different emission components
        the first 18 are the astrophysical rings of Pohl et al.

    Returns
    -------
    float
        zero if you respect the priors, or a big negative number if you violate the priors
    """
    argL, argR = (theta-(pmin[:len(theta)]-1))/1, ((pmax[:len(theta)]+1)-theta)/1
    # sigL, sigR = 1./(1.+jnp.exp(-argL)), 1./(1.+jnp.exp(-argR))
    # return jnp.sum(jnp.log(sigL)) + jnp.sum(jnp.log(sigR))
    return jnp.sum(argL - jsc.xlog1py(1, jnp.exp(argL)) + argR - jsc.xlog1py(1, jnp.exp(argR)))
jjlnprior_smooth = jit(jlnprior_smooth)

#the sum of the hard prior-enforcing function and the log likelihood -- numpyro prefers this
def jlnprob_hard(theta, bin_no=0, ex_num=1):
    """
    function that returns a log-likelihood that the data is described by a model of the gamma-ray sky
    
    Parameters
    ----------
    theta: vector
        log10 of the normalizations of the different emission components
        the first 18 are the astrophysical rings of Pohl et al.
    
    bin_no: int
        the energy bin number
    
    ex_num: int
        a way of selecting the excess to add to the astrophysical rings
        0 corresponds to no excess
        1 corresponds to DM
        2 is the boxy bulge
        3 is the x-shaped bulge
        4 is the 'stellar bulge', boxy plus nuclear with a certain ratio constrained in a certain way, aka dm_64
        5 allows both the boxy bulge and the x-shaped bulge to be free independently
    
    Returns
    -------
    float
        the log-likelihood _from the UNconstrained space_ that the data is described by a model of the gamma-ray sky
    """
    return jjlnprior_hard(theta) + jjlnlike(theta, bin_no, ex_num)
jjlnprob_hard = jit(jlnprob_hard, static_argnums=(1,2))


# the sum of the smooth prior-enforcing function and the log likelihood -- mchmc prefers this
def jlnprob_smooth(theta, bin_no = 0, ex_num = 1):
    """
    function that returns a log-likelihood that the data is described by a model of the gamma-ray sky

    Parameters
    ----------
    theta: vector
        log10 of the normalizations of the different emission components
        the first 18 are the astrophysical rings of Pohl et al.

    bin_no: int
        the energy bin number

    ex_num: int
        a way of selecting the excess to add to the astrophysical rings
        0 corresponds to no excess
        1 corresponds to DM
        2 is the boxy bulge
        3 is the x-shaped bulge
        4 is the 'stellar bulge', boxy plus nuclear with a certain ratio constrained in a certain way, aka dm_64
        5 allows both the boxy bulge and the x-shaped bulge to be free independently

    Returns
    -------
    float
        the log-likelihood _from the UNconstrained space_ that the data is described by a model of the gamma-ray sky
    """
    return jjlnprior_smooth(theta) + jjlnlike(theta, bin_no, ex_num)
jjlnprob_smooth = jit(jlnprob_smooth, static_argnums = (1, 2))