from setuptools import setup, find_packages

setup(
    name='debacl',
    version='1.0',
    description='DEnsity-BAsed CLustering',
    url='https://github.com/CoAxLab/DeBaCl',
    author='Brian Kent',
    author_email='bpkent@gmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering'
        ],
    packages=find_packages(),
    install_requires=["prettytable"]
)
