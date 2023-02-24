import sys
import gcepy.lowdim_model as lm
import dynesty as dn
import numpyro as nr
from jax import jit
from jax import numpy as jnp
import numpy as np
import jax.random as jr

def ptform(x, dim):
    return x*(lm.pmax[:dim]-lm.pmin[:dim])+lm.pmin[:dim]
jptform = jit(ptform, static_argnums=(1,))

EBIN, EXNUM = int(sys.argv[1]), int(sys.argv[2])

modparams = [4, 5, 5, 5, 5, 6]
PDIM = modparams[EXNUM]

sampler_dn = dn.NestedSampler(lambda x: lm.jjlnlike(x, EBIN, ex_num=EXNUM), lambda x: jptform(x, PDIM), PDIM)

sampler_dn.run_nested(dlogz=1)

sampler_nr = nr.infer.MCMC(nr.infer.NUTS(potential_fn = lambda t: -lm.jjlnprob(t, bin_no=EBIN, ex_num=EXNUM)), num_samples=10000, num_warmup=100, jit_model_args=True, chain_method='vectorized')

rng_key = jr.PRNGKey(0)

sampler_nr.run(rng_key, init_params = sampler_dn.results.samples[-1])

samples_nr = sampler_nr.get_samples()

lls_nr = jnp.array([lm.jjlnprob(x, bin_no=EBIN, ex_num=EXNUM) for x in samples_nr])

sampler_dn.results.logl[-1], float(lls_nr.max()), float(-sampler_nr.last_state.potential_energy), int(lls_nr.argmax())

np.savetxt("results/ebin" + str(EBIN) + "exnum" + str(EXNUM) + "Lsummary.txt", np.array([sampler_dn.results.logz[-1], sampler_dn.results.logl[-1], float(lls_nr.max()), float(-sampler_nr.last_state.potential_energy)]))