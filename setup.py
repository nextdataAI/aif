from setuptools import find_packages, setup

README = open("README.md").read()

setup(
    name='nextdataAI',
    packages=find_packages(),
    version='0.0.16',
    long_description_content_type="text/markdown",
    url="https://github.com/nextdataAI/aif",
    long_description=README,
    author_email='chuckpaul98@icloud.com, s.zanoni@studenti.unipi.it',
    install_requires=['wandb', 'numpy', 'matplotlib', 'scipy','gym','minihack', 'nle','keras','tensorflow','torch','Pillow','imageio','pandas'],
    setup_requires=['pytest-runner'],
    author='Paul Magos & Stefano Zanoni',
    python_requires='>=3.9',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)