import setuptools

setuptools.setup(
    name="gcepy",
    version="0.1",
    author="Samuel D. McDermott",
    author_email="samueldmcdermott@gmail.com",
    description="likelihoods and samplers for the Galactic center excess",
    packages=["gcepy"],
    url="https://github.com/samueldmcdermott/gcepy",
    install_requires=['setuptools',"jax", "dynesty", "numpyro"],
    package_data={
        "gcepy": ["inputs/utils/*.npy", "inputs/templates_lowdim/*.npy","inputs/excesses/*.npy"],
    }
)