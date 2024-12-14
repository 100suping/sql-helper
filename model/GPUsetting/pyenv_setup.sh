#!/bin/bash

# pyenv 설치
curl https://pyenv.run | bash && echo "pyenv installed."

# ~/.bashrc 파일에 환경 변수 추가
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# 환경 변수 적용
source ~/.bashrc && echo ".bashrc updated with pyenv variables."
exec $SHELL