from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'
def getRequirements()->List[str]:
    with open('requirements.txt') as f:
        requirements = f.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements
setup(
    name='mlproject',
    version='0.1',
    packages=find_packages(),
    author='Panshul Jindal',
    author_email="panshuljindal@gmail.com",
    install_requires=getRequirements()
)
