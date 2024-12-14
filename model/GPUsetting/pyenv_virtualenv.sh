#!/bin/bash

# Python 버전 설치 (사용자가 직접 버전을 입력하도록 대화형으로 구현)
read -p "Enter the version of Python you want to install (e.g., 3.11.x): " PYTHON_VERSION
pyenv install $PYTHON_VERSION && echo "Python version $PYTHON_VERSION installed."
pyenv init
# 원하는 버전으로 설정
pyenv shell $PYTHON_VERSION && echo "Python version set to $PYTHON_VERSION."

# 원하는 이름의 virtualenv 생성
read -p "Enter the name for the virtual environment: " VENV_NAME
pyenv virtualenv $PYTHON_VERSION "$VENV_NAME" && echo "Virtual environment '$VENV_NAME' created."

# 가상환경 활성화
pyenv activate "$VENV_NAME" && echo "Virtual environment '$VENV_NAME' activated."