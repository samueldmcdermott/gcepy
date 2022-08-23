# gce
Code for high-dimensional sampling for testing models of the Galactic center excess (GCE).

This code relies on jax for all basic definitions, building up to a log-likelihood for the model given the data.
The repository provides an ipynb demonstrating how to use two high-dimensional samplers, dynesty and numpyro, to derive constraints on the parameters.
Because the problem is inherently parallelizable, two example scripts are provided to facilitate batch computing with these tools.

Models and data are available on request.
Permission is necessary from the first author of the relevant paper.
