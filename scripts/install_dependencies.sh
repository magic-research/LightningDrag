#!/bin/bash

echo "Installing dependencies ..."

pip3 install numpy==1.23.5
pip3 install transformers==4.36.2
pip3 install imageio==2.27.0
pip3 install imageio-ffmpeg==0.4.9
pip3 install omegaconf==2.3.0
pip3 install gradio==3.50.2
pip3 install einops==0.7.0
pip3 install einops-exts==0.0.4
pip3 install mmcv==1.7.1
pip3 install diffusers==0.29.2
pip3 install peft==0.10.0
pip3 install safetensors
pip3 install pytorch_lightning
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118