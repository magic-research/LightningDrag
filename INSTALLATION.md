# Installation

## 1. Setup environment
```
conda create --name lightning-drag python=3.9 pip
conda activate lightning-drag
bash scripts/install_dependencies.sh
```

## 2. Download pretrained models
Download the following models and place them under `./checkpoints`
<!-- - [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) -->
- [dreamshaper-8-inpainting](https://huggingface.co/Lykon/dreamshaper-8-inpainting)
- [lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
- [IP-Adapter](https://huggingface.co/h94/IP-Adapter/tree/main/models)
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [lightning-drag-sd15](https://huggingface.co/LightningDrag/lightning-drag-sd15)

Place the checkpoints as follows:
```
└── checkpoints
    ├── dreamshaper-8-inpainting
    ├── lcm-lora-sdv1-5
    │   └── pytorch_lora_weights.safetensors
    ├── sd-vae-ft-ema
    │   ├── config.json
    │   ├── diffusion_pytorch_model.bin
    │   └── diffusion_pytorch_model.safetensors
    ├── IP-Adapter/models
    │   ├── image_encoder
    │   └── ip-adapter_sd15.bin
    └── lightning-drag-sd15
        ├── appearance_encoder
        │   ├── config.json
        │   └── diffusion_pytorch_model.safetensors
        ├── point_embedding
        │   └── point_embedding.pt
        └── lightning-drag-sd15-attn.bin
```

