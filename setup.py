from distutils.core import setup

setup(
    name='debacl',
    version='0.3.0',
    author='Brian P. Kent',
    author_email='bpkent@gmail.com',
    packages=['debacl', 'debacl.test'],
    scripts=['examples/1d_gauss_demo.py', 'examples/crater_demo.py'],
    url='https://github.com/CoAxLab/DeBaCl',
    license='BSD',
    description='DEnsity-BAsed CLustering',
    long_description=open('README.md').read(),
    install_requires=[
        "prettytable"
    ],
)
