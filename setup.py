from typing import List
import setuptools
from pathlib import Path

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def get_install_requires() -> List[str]:
    """Returns requirements.txt parsed to a list"""
    fname = Path(__file__).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
    return targets

setuptools.setup(
    name='dionn',
    version='1.5.0',
    author='Juan Zamora, Sebastian Vegas, Kerlyns Martínez, Daira Velandia, Sebastián Jara, Pascal Sigel',
    author_email='',
    description='Detection Intra-class Outliers with Neural Networks (DIONN) algorithm',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/juanzamorai/intracluster-filtering',
    license='MIT',
    packages=['dionn', 'utils'],
    install_requires=get_install_requires(),
)
