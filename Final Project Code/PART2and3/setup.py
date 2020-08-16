
import setuptools

VERSION = '0.0.6'

with open('README.md', 'r') as file_object:
  LONG_DESCRIPTION = file_object.read()

setuptools.setup(
    name='uisrnn',
    version=VERSION,
    author='Quan Wang',
    author_email='quanw@google.com',
    description='Unbounded Interleaved-State Recurrent Neural Network',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/google/uis-rnn',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
)
