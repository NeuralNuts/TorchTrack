from setuptools import find_packages, setup

setup(
    name="track_ml",
    packages=find_packages(include=["track_ml"]),
    version="0.1.0",
    description="TrackML takes your models data and training data then send it to the TrackML web application",
    author="NeuralNuts",
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='test',
)
