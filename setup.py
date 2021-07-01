from setuptools import setup, find_packages
import os


def get_requirements(req_file):
    reqs = []
    with open(req_file, "r") as fp:
        for line in fp.readlines():
            if line.startswith("#") or line.strip() == "":
                continue
            else:
                reqs.append(line.strip())
    return reqs


# A trick from https://github.com/jina-ai/jina/blob/79b302c93b01689e82cf4b52f46522eb7497c404/setup.py#L20
pkg_name = "vila"
libinfo_py = os.path.join("src", pkg_name, "__init__.py")
libinfo_content = open(libinfo_py, "r", encoding="utf8").readlines()
version_line = [l.strip() for l in libinfo_content if l.startswith("__version__")][0]
exec(version_line)  # gives __version__

setup(
    name=pkg_name,
    version=__version__,
    author="Zejiang Shen",
    license="Apache-2.0",
    url="https://github.com/allenai/vila",
    package_dir={"": "src"},
    packages=find_packages("src"),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=get_requirements("requirements.txt"),
)
