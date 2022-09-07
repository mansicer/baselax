import itertools
from setuptools import setup, find_namespace_packages

BASELAX_VERSION = '0.0.1'

extras = {
    "nni": ["nni"],
}

extras["all"] = list(
    set(itertools.chain.from_iterable(map(lambda group: extras[group], extras.keys())))
)

requires = [
    "gym",
    "optax",
    "rlax",
    "stable-baselines3",
    "UtilsRL",
    "dm-haiku",
]

setup(
    name="baselax",
    version=BASELAX_VERSION,
    packages=find_namespace_packages(include=["baselax.*"]),
    install_requires=requires,
    extras_require=extras,
)
