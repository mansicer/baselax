import re
import itertools
from setuptools import setup, find_namespace_packages

with open("baselax/version.py") as file:
    full_version = file.read()
    assert (
        re.match(r'VERSION = "\d\.\d+\.\d+"\n', full_version).group(0) == full_version
    ), f"Unexpected version: {full_version}"
    BASELAX_VERSION = re.search(r"\d\.\d+\.\d+", full_version).group(0)

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
