#!/bin/bash

pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip && pip install unsloth && echo "Backend setup complete!"