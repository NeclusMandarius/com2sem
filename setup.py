from setuptools import setup

setup(
    name='com2sem',
    version='0.0.1',
    packages=['com2sem'],
    install_requirements=["scikit-learn", "numpy", "PyYAML"],
    url='https://github.com/NeclusMandarius/com2sem.git',
    license='MIT License',
    author='Sebastian Schmidt',
    author_email='nm.sebastian.schmidt@gmail.com',
    description='Package providing an implementation for decision tree based Word Embedding'
)
