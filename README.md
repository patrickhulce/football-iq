# football-iq

Gathering intelligence about football.

## Setup

conda env create -n football-iq -f environment.yml
conda activate football-iq

# Miscellaneous

## Modifying Python Deps

```sh
conda activate football-iq
conda install pytorch torchvision -c pytorch # Or whatever deps you're adding.
conda env export --from-history > environment.yml # Update and commit the environment file.
```

## Prerequisites

You'll need a working conda environment.

```sh
brew install miniconda # Install miniconda for managing python environments
conda init "$(basename "${SHELL}")" # Setup conda in your shell
conda config --set auto_activate_base false # (If you don't normally program in python), disable it from autoloading.
```

## How This Repo Was Set Up

```sh
conda create -n football-iq python=3.11
conda activate football-iq
conda install pytorch torchvision torchaudio -c pytorch -c conda-forge
conda env export > environment.yml
```
