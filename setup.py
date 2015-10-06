from distutils.core import setup

setup(
    name='debacl',
    version='0.3.0',
    author='Brian P. Kent',
    author_email='bpkent@gmail.com',
    packages=['debacl', 'debacl.test'],
    scripts=['bin/gauss_demo.py'],
    url='https://github.com/CoAxLab/DeBaCl',
    license='LICENSE.txt',
    description='Density-Based Clustering',
    long_description=open('README.md').read(),
    install_requires=[
        "networkx",
        "numpy",
        "prettytable"
    ],
)
