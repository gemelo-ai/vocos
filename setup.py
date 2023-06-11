import io
import os

from setuptools import find_packages, setup

for line in open("vocos/__init__.py"):
    line = line.strip()
    if "__version__" in line:
        context = {}
        exec(line, context)
        VERSION = context["__version__"]


def read(*paths, **kwargs):
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths), encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [line.strip() for line in read(path).split("\n") if not line.startswith(('"', "#", "-", "git+"))]


setup(
    name="vocos",
    version=VERSION,
    author="Hubert Siuzdak",
    author_email="huberts@charactr.com",
    description="Fourier-based neural vocoder for high-quality audio synthesis",
    url="https://github.com/charactr-platform/vocos",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    extras_require={"train": read_requirements("requirements-train.txt")},
)
