<p align="center">
  <h1 align="center">
  LightningDrag: Lightning Fast and Accurate Drag-based Image Editing Emerging from Videos
  </h1>
  <p align="center">
    <a href="https://yujun-shi.github.io/"><strong>Yujun Shi</strong></a><sup>*</sup>
    &nbsp;&nbsp;
    <a href="https://scholar.google.com.sg/citations?user=8gm-CYYAAAAJ&hl=en"><strong>Jun Hao Liew</strong></a><sup>*^</sup>
    &nbsp;&nbsp;
    <a href="https://hanshuyan.github.io/"><strong>Hanshu Yan</strong></a>
    &nbsp;&nbsp;
    <a href="https://vyftan.github.io/"><strong>Vincent Y. F. Tan</strong></a>
    &nbsp;&nbsp;
    <a href="https://sites.google.com/site/jshfeng/home"><strong>Jiashi Feng</strong></a>
    <br>
    <b>National University of Singapore &nbsp; | &nbsp;  ByteDance</b>
  </p>

  <div align="center">
      <sup>*&nbsp;</sup>Equal Contributions&nbsp;&nbsp;&nbsp;&nbsp;<sup>^&nbsp;</sup>Project Lead
  </div>

  <p align="center">
    <a href="https://arxiv.org/abs/2405.13722"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2405.13722-b31b1b.svg"></a>
    <a href="https://lightning-drag.github.io"><img alt='page' src="https://img.shields.io/badge/Project-Website-orange"></a>
    <a href="https://twitter.com/YujunPeiyangShi"><img alt='Twitter' src="https://img.shields.io/twitter/follow/YujunPeiyangShi?label=%40YujunPeiyangShi"></a>
    <a href="https://twitter.com/jhliew91"><img alt='Twitter' src="https://img.shields.io/twitter/follow/jhliew91?label=%40jhliew91"></a>
  </p>

  <div align="center">
    <h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.
  </div> 

  <div align="center">
    <img src="./assets/demo_shortest.gif", width="700">
  </div>
  <br>
</p>

## Disclaimer
This is a research project, NOT a commercial product. Users are granted the freedom to create images using this tool, but they are expected to comply with local laws and utilize it in a responsible manner. The developers do NOT assume any responsibility for potential misuse by users.

## Update
- [2024.09.17] Release inference code and model.

## Installation
See [Installation](INSTALLATION.md) for installation.

## Gradio demo
Run gradio locally by
```
python3 drag_ui.py \
    --base_sd_path="checkpoints/dreamshaper-8-inpainting/" \
    --vae_path="checkpoints/sd-vae-ft-mse/" \
    --ip_adapter_path="checkpoints/IP-Adapter/models/" \
    --lightning_drag_model_path="checkpoints/lightning-drag-sd15" \
    --lcm_lora_path="checkpoints/lcm-lora-sdv1-5"
```
Please refer to the [GIF](assets/demo_shortest.gif) above for step-by-step demo on how to use this UI.

**IMPORTANT NOTES**
- Since [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) is no longer available, we replace the inpainting checkpoint with [`Lykon/dreamshaper-8-inpainting`](https://huggingface.co/Lykon/dreamshaper-8-inpainting) without retraining or finetuning. Although it works, the results may not look the same as the one in the paper.
- To reproduce the results from the paper, you may need to replace `base_sd_path` with your own `stable-diffusion-v1-5` checkpoint.

## Qualitative Results Gallery
<div align="center">
  <h4 align="center">Single-round Dragging</p>
  <img src="./assets/single_round.png", width="85%">
  <h4 align="center">Multi-round Dragging</p>
  <img src="./assets/multi_round.png", width="85%">
</div>

## Contact
For any questions on this project, please contact [Yujun](https://yujun-shi.github.io/) (shi.yujun@u.nus.edu) and [Jun Hao](https://scholar.google.com.sg/citations?user=8gm-CYYAAAAJ&hl=en) (junhao.liew@bytedance.com)

## BibTeX
If you find our repo helpful, please consider leaving a star or cite our paper :)
```bibtex
@article{shi2024lightningdrag,
         title={LightningDrag: Lightning Fast and Accurate Drag-based Image Editing Emerging from Videos},
         author={Shi, Yujun and Liew, Jun Hao, and Yan, Hanshu and Tan, Vincent YF and Feng, Jiashi},
         journal={arXiv preprint arXiv:2405.13722},
         year={2024}
}
```

## Acknowledgement
Source image samples are collected from [unsplash](https://unsplash.com/), [pexels](https://www.pexels.com/zh-cn/), [pixabay](https://pixabay.com/). Also, a huge shout-out to all the amazing open source diffusion models, libraries, and technical reports.
