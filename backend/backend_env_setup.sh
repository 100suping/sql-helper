#!/bin/bash
# frontend_env_setup.sh

# Set default values
PYTHON_VERSION="3.11.8"
VENV_NAME="backend"

# 1. Install dependencies
sudo apt-get update && sudo apt-get install -y curl vim make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev

# 2. Install pyenv
curl https://pyenv.run | bash

# 3. Configure environment
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# 4. Load environment
source ~/.bashrc

# 5. Install Python and create virtualenv
pyenv install $PYTHON_VERSION
pyenv shell $PYTHON_VERSION
pyenv virtualenv $PYTHON_VERSION $VENV_NAME

echo "Setup complete. Please run: 'pyenv activate backend' to start using the environment"