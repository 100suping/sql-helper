#!/bin/bash


# pyenv 종속성 설치
echo "Installing dependencies for pyenv..."
bash pyenv_dependencies.sh

# pyenv 설치 및 설정
echo "Setting up pyenv..."
bash pyenv_setup.sh

# Python 버전 및 가상환경 설정
echo "Installing Python and setting up virtual environment..."
bash pyenv_virtualenv.sh