from setuptools import setup, find_packages

setup(
    name="gcepy",
    version="0.1",
    author="Samuel D. McDermott",
    author_email="samueldmcdermott@gmail.com",
    description="likelihoods and samplers for the Galactic center excess",
    packages=find_packages(),
    url="https://github.com/samueldmcdermott/gcepy",
    install_requires=['setuptools',"jax", "dynesty", "numpyro"],
    package_data={
        "gcepy": ["inputs/utils/*.npy", "inputs/templates_lowdim/*.npy", "inputs/templates_highdim/*.npy",
                  "inputs/excesses/*.npy"],
    }
)