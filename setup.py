from setuptools import setup, find_packages
from os.path import join, dirname

setup(
    name='point_cloud_library',
    version='1.0',
    author='Artem',
    author_email='artl2sch@gmail.com',
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.txt')).read(),
)
