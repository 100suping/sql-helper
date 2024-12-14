#!/bin/bash

# CUDA 및 NVIDIA 드라이버 설치
sudo apt-get update && echo "System update complete."
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall && echo "NVIDIA driver installed."

# 시스템 재부팅 (주의: 스크립트 재실행 필요)
sudo reboot