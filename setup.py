from setuptools import find_packages, setup

install_requires = [
    "datasets~=2.14.5",  # HuggingFace open-source datasets
    "dill~=0.3.6",  # serializing/deserializing models
    "faker~=19.6.1",  # fake/random text generation
    "fastparquet~=2023.7.0",  # parquet datasets
    "Flask~=2.3.3",  # web framework
    "matplotlib~=3.7.2",  # graphs and plots
    "numpy~=1.25.1",  # data structures
    "pandas~=2.0.3",  # data processing
    "pyarrow~=12.0.1",  # parquet datasets
    "pycountry~=22.3.5",  # language names from codes
    "pytest~=7.4.2",  # unit testing
    "scikit-learn~=1.3.0",  # for building the language classification model
    "speechbrain~=0.5.15",  # used for speech language recognition model
    "speechrecognition~=3.10.0",  # used for speech transcription model
    "tensorflow~=2.11.1",  # for building the handwriting recognition model
    "torchaudio~=2.0.2"  # audio processing
    "zstandard~=0.21.0",  # data compression
]

setup(name="lingodect", version="1.0", packages=find_packages())
