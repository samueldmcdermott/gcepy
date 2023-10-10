# gcepy
Code for sampling for testing models of the Galactic center excess (GCE).

This code relies on `jax` for all basic definitions, building up to a log-likelihood for the model given the data.

There are separate modules for a "low-dimensional" (astrophysically motivated) 5-parameter model based on [2112.09706](http://arxiv.org/abs/2112.09706) and a "high-dimensional" (ring-based) 19-parameter model based on [2209.00006](https://arxiv.org/abs/2209.00006).
Models and data related to [2112.09706](http://arxiv.org/abs/2112.09706) are included [here](gcepy/inputs/templates_lowdim).
Model files based on the ring-based models of [2209.00006](https://arxiv.org/abs/2209.00006) (and Pohl et al.) are available on request.

## installation and usage

To install, clone the repository and run `python -m pip install .` as normal.

To load and use the "low-dimensional" model the preferred method is
```
import gcepy
from jax import numpy as jnp
gcepy.lnlike('low', jnp.ones(5))
```
or the following is equivalent:
```
import gcepy
from jax import numpy as jnp
gcepy._lm.jjlnprob(jnp.ones(5))
```

See the included ipython notebook for a more involved example, including both `dynesty` and `numpyro` runs.

## citation
If you use this module in your research please add a footnote to this repository and cite [2112.09706](http://arxiv.org/abs/2112.09706) and [2209.00006](https://arxiv.org/abs/2209.00006) and any other relevant work.