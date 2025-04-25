from setuptools import setup, find_packages

setup(
    name="ml-lib",
    version="0.1.0",
    description="A machine learning library implementing core algorithms",
    author="Nitesh",
    author_email="n9106822@gmail.com",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'scikit-learn>=0.24.0'
    ],
    python_requires='>=3.6',
)