# gcepy
Code for sampling for testing models of the Galactic center excess (GCE).

This code relies on jax for all basic definitions, building up to a log-likelihood for the model given the data.
The repository provides an ipynb demonstrating how to use two high-dimensional samplers, dynesty and numpyro, to derive constraints on the parameters.

There are separate modules for a "low-dimensional" (astrophysically motivated) 5-parameter model based on 2112.09706 and a "high-dimensional" (ring-based) 19-parameter model based on 2209.abcde.
Models and data related to 2112.09706 are included here.
Model files based on 2209.abcde are available on request.

To install, clone the repository and run ```python setup.py install``` as normal.

To load and use the "low-dimensional" model (which will work from any shell after installation), do `import gcepy.lowdim_model as lm` and then e.g. `lm.jjlnprob(lm.jnp.ones(5))`
