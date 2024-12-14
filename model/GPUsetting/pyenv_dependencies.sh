#!/bin/bash

# curl, vim 설치
sudo apt-get update && sudo apt-get install -y curl vim && echo "curl, vim installed."

# 종속성 설치 (pyenv 설정 전에 필요)
sudo apt-get update
sudo apt-get install -y \
    make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    wget llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev && \
    echo "All dependencies installed."