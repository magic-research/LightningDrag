# Installation

## 1. Setup environment
```
conda create --name lightning-drag python=3.9 pip
conda activate lightning-drag
pip3 install -r requirements.txt
```

## 2. Download pretrained models
Download the following models and place them under `./checkpoints`
<!-- - [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) -->
- [dreamshaper-8-inpainting](https://huggingface.co/Lykon/dreamshaper-8-inpainting) (refer to [IMPORTANT NOTES](#important-notes))
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

## IMPORTANT NOTES
- Since [`runwayml/stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting) is no longer available, we replace the inpainting checkpoint with [`Lykon/dreamshaper-8-inpainting`](https://huggingface.co/Lykon/dreamshaper-8-inpainting) without retraining or finetuning. Although it works, the results may not look the same as the one in the paper.
- To reproduce the results from the paper, you may need to replace the inpainting checkpoint with your own `stable-diffusion-inpainting` checkpoint.