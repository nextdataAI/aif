from setuptools import find_packages, setup

README = open("README.md").read()

setup(
    name='nextdataAI',
    packages=find_packages(),
    version='0.0.2',
    long_description_content_type="text/markdown",
    url="https://github.com/nextdataAI/aif",
    long_description=README,
    author_email='chuckpaul98@icloud.com',
    install_requires=['numpy', 'matplotlib', 'pandas', 'tqdm', 'wandb'],
    setup_requires=['pytest-runner'],
    author='Paul Magos & Stefano Zanoni',
    python_requires='>=3.9',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)