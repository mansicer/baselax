import itertools
from setuptools import setup, find_namespace_packages

with open("baselax/version.txt", "r", encoding="utf-8") as f:
    BASELAX_VERSION = f.read().strip()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requires = f.read().splitlines()

extras = {
    "sb3": ["stable-baselines3"],
}

extras["all"] = list(
    set(itertools.chain.from_iterable(map(lambda group: extras[group], extras.keys())))
)

setup(
    name="baselax",
    version=BASELAX_VERSION,
    packages=find_namespace_packages(include=["baselax.*"]),
    install_requires=requires,
    extras_require=extras,
)
