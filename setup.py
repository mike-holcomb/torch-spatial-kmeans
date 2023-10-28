from setuptools import setup, find_packages

setup(
    name='torch_spatial_kmeans',
    version='0.1.0',
    author='Mike Holcomb',
    url='https://github.com/mike-holcomb/torch-spatial-kmeans',
    description='A PyTorch implementation of spatial K-means clustering.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='k-means clustering pytorch spatial',
)
