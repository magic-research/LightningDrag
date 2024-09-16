# # running without LCM
# export no_proxy="localhost, 127.0.0.1"; python3 drag_ui.py \
#     --base_sd_path="checkpoints/stable-diffusion-v1-5/" \
#     --vae_path="checkpoints/sd-vae-ft-mse/" \
#     --ip_adapter_path="checkpoints/IP-Adapter/models/" \
#     --lightning_drag_model_path="checkpoints/lightning-drag-sd15"
    
# running with LCM
export no_proxy="localhost, 127.0.0.1"; python3 drag_ui.py \
    --base_sd_path="checkpoints/dreamshaper-8-inpainting/" \
    --vae_path="checkpoints/sd-vae-ft-mse/" \
    --ip_adapter_path="checkpoints/IP-Adapter/models/" \
    --lightning_drag_model_path="checkpoints/lightning-drag-sd15" \
    --lcm_lora_path="checkpoints/lcm-lora-sdv1-5"
