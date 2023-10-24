from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    rq = []
    with open(file_path) as file_obj:
        rq = file_obj.readlines()
        rq = [req.replace('\n', '') for req in rq]

        return rq


setup(
    name='DiamondPrice',
    version='0.0.1',
    author='Shivansh',
    author_email='xyz@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)
