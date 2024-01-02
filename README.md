# football-iq

Gathering intelligence about football.

# Development

## Modifying Python Deps

```sh
conda activate football-iq
conda install pytorch torchvision -c pytorch # Or whatever deps you're adding.
conda env export > environment.yml # Update and commit the environment file.
```

## Setup

### macOS

You'll need a working conda environment.

```sh
brew install miniconda # Install miniconda for managing python environments
conda init "$(basename "${SHELL}")" # Setup conda in your shell
conda config --set auto_activate_base false # (If you don't normally program in python), disable it from autoloading.
conda env create -n football-iq -f environment.yml # Create the environment.
conda activate football-iq # Use it.
```

### Windows

In PowerShell (Admin):

```powershell
choco install anaconda3 git # Install Anaconda and Git on your machine
# Manually setup Windows terminal to use the following for Git-Bash
C:\Tools\Anaconda3\python.exe C:\Tools\Anaconda3\cwp.py "C:%HOMEPATH%\.conda\envs\football-iq" "C:\Program Files\Git\bin\bash.exe" --login -i --

```

In Git-Bash:

```sh
export PATH="/c/tools/Anaconda3/Scripts:$PATH" # Add anaconda to your path for this session.
conda env create -n football-iq -f C:\path\to\repo\environment.yml # Create the environment.
activate football-iq # Use it.
```

## How This Repo Was Set Up

```sh
conda create -n football-iq python=3.11
conda activate football-iq
conda install pytorch torchvision torchaudio -c pytorch -c conda-forge
conda env export > environment.yml
```
