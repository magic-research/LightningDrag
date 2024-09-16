# *************************************************************************
# Copyright (2024) Bytedance Inc.
#
# Copyright (2024) LightningDrag Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

import os
import cv2
import math
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from collections import OrderedDict

import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
from safetensors.torch import load_file

from models.ip_adapter import ImageProjModel
from models.appearance_encoder import AppearanceEncoderModel
from models.point_embedding import PointEmbeddingModel
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from utils.utils import import_model_class_from_model_name_or_path

from pytorch_lightning import seed_everything
from pipeline.lightningdrag_pipeline import LightningDragPipeline

class LightningDragUI:

    def __init__(
        self,
        base_sd_path,
        vae_path,
        ip_adapter_path,
        lightning_drag_path,
        lcm_lora_path=None,
    ):
        self.__init_model(
            base_sd_path,
            vae_path,
            ip_adapter_path,
            lightning_drag_path,
            lcm_lora_path,
        )

    def __init_model(
        self,
        base_sd_path,
        vae_path,
        ip_adapter_path,
        lightning_drag_path,
        lcm_lora_path,
    ):
        self.base_sd_path = base_sd_path
        self.vae_path = vae_path

        device = "cuda"
        dtype = torch.float16

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_sd_path,
            subfolder="tokenizer",
            use_fast=False,
        )
        # Load text encoder
        text_encoder_cls = import_model_class_from_model_name_or_path(base_sd_path, revision=None)
        text_encoder = text_encoder_cls.from_pretrained(
            base_sd_path, subfolder="text_encoder"
        )
        # Load vae
        if vae_path == "default":
            vae = AutoencoderKL.from_pretrained(base_sd_path)
        else:
            vae = AutoencoderKL.from_pretrained(vae_path)

        # Load inpaint unet
        # unet = UNet2DConditionModel.from_config(
        #     lightning_drag_path, subfolder="unet"
        # )
        unet = UNet2DConditionModel.from_pretrained(
            base_sd_path, subfolder="unet"
        )
        config = unet.config
        config['in_channels'] = 4
        appearance_encoder = AppearanceEncoderModel.from_config(config)

        noise_scheduler = DDIMScheduler.from_pretrained(base_sd_path,
                                                    subfolder="scheduler")

        # Load image encoder
        image_encoder_path = os.path.join(ip_adapter_path, "image_encoder")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
        clip_image_processor = CLIPImageProcessor()
        # Load ip-adapter image proj model
        image_proj_model = ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=image_encoder.config.projection_dim,
            clip_extra_context_tokens=4, # HACK: hard coded to be 4 here, as we are using the normal ip adapter
        )
        ip_ckpt_path = os.path.join(ip_adapter_path, "ip-adapter_sd15.bin")
        ip_state_dict = torch.load(ip_ckpt_path, map_location="cpu", weights_only=True)
        image_proj_model.load_state_dict(ip_state_dict["image_proj"])

        point_embedding = PointEmbeddingModel(embed_dim=16)

        unet.to(device).to(dtype)
        vae.to(device).to(dtype)
        text_encoder.to(device).to(dtype)
        appearance_encoder.to(device).to(dtype)
        point_embedding = point_embedding.to(device).to(dtype)
        image_encoder = image_encoder.to(device).to(dtype)
        image_proj_model = image_proj_model.to(device).to(dtype)

        pipe = LightningDragPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            appearance_encoder=appearance_encoder,
            scheduler=noise_scheduler,
            feature_extractor=clip_image_processor,
            image_encoder=image_encoder,
            point_embedding=point_embedding,
            safety_checker=None,
            fusion_blocks="full",
            initialize_attn_processor=True,
            use_norm_attn_processor=True,
            initialize_ip_attn_processor=True,
            image_proj_model=image_proj_model,
        )
        self.pipe = pipe
        self.load_model(lightning_drag_path, base_sd_path, ip_state_dict)

        if lcm_lora_path is not None:
            self.pipe.load_lora_weights(lcm_lora_path)
            self.pipe.fuse_lora()
            self.pipe.scheduler = LCMScheduler.from_pretrained(base_sd_path, subfolder="scheduler")

        self.pipe = self.pipe.to(device).to(dtype)

    # load lightning drag model
    def load_model(self, lightning_drag_path, base_sd_path, ip_state_dict):

        # Load weights for attn_processors, including those from IP-Adapter
        attn_processors = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        state_dict = torch.load(
            os.path.join(lightning_drag_path, "lightning-drag-sd15-attn.bin")
        )
        state_dict.update(ip_state_dict["ip_adapter"])
        attn_processors.load_state_dict(state_dict)

        # Load appearance encoder
        appearance_state_dict = load_file(
            os.path.join(lightning_drag_path,
                         "appearance_encoder/diffusion_pytorch_model.safetensors")
        )
        self.pipe.appearance_encoder.load_state_dict(appearance_state_dict)

        # Load point embedding
        point_embedding_state_dict = torch.load(
            os.path.join(lightning_drag_path, "point_embedding/point_embedding.pt")
        )
        self.pipe.point_embedding.load_state_dict(point_embedding_state_dict)

    # -------------- general UI functionality --------------
    def clear_all(self, length=480):
        display_size = int(length)
        return gr.Image(value=None, height=display_size, interactive=True), \
            gr.Image(value=None, height=display_size, interactive=False), \
            gr.Gallery(
                value=None,
                label="Dragged Images",
                show_label=True,
                elem_id="gallery",
                columns=[2],
                rows=[2],
                object_fit="contain",
                height=2*length,
                width=2*length,
                ), \
            [], None, None

    def select_image(self, images, img_id=0, length=480):
        display_size = int(length)
        return images[img_id]["name"], \
            images[img_id]["name"], \
            gr.Gallery(
                value=None,
                label="Dragged Images",
                show_label=True,
                elem_id="gallery",
                columns=[2],
                rows=[2],
                object_fit="contain",
                height=2*length,
                width=2*length,
                ), \
            [], \
            images[img_id]["name"], \
            None

    def __mask_image(
                self,
                image,
                mask,
                color=[255,0,0],
                alpha=0.5):
        """ Overlay mask on image for visualization purpose. 
        Args:
            image (H, W, 3) or (H, W): input image
            mask (H, W): mask to be overlaid
            color: the color of overlaid mask
            alpha: the transparency of the mask
        """
        out = deepcopy(image)
        img = deepcopy(image)
        img[mask == 1] = color
        out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
        return out

    def __resize_divisible_by_64(self, image, size=512):
        """Resize image such that its height and width are divisible by 64.
        """
        if isinstance(image, Image.Image):
            w, h = image.size
        elif isinstance(image, np.ndarray):
            h, w, _ = image.shape
        else:
            raise NotImplementedError
        ho, wo = h, w

        resize_coef = math.sqrt(size*size/(h*w))
        resize_h, resize_w = h*resize_coef, w*resize_coef
        h, w = int(np.round(resize_h / 64.0)) * 64, int(np.round(resize_w / 64.0)) * 64

        return h, w, ho, wo

    def store_img(self, img, length=512):
        image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
        height,width,_ = image.shape
        image = Image.fromarray(image)
        image = exif_transpose(image)

        # resize to short_side=512
        height, width, _, _ = self.__resize_divisible_by_64(
            image,
            size=512,
        )

        image = image.resize((width, height), PIL.Image.LANCZOS)
        mask  = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        image = np.array(image)

        if mask.sum() > 0:
            mask = np.uint8(mask > 0)
            masked_img = self.__mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
        else:
            masked_img = image.copy()
        # when new image is uploaded, `selected_points` should be empty
        return image, [], gr.Image(value=masked_img, interactive=True), mask

    # user click the image to get points, and show the points on the image
    def get_points(
                self,
                img,
                sel_pix,
                evt: gr.SelectData):
        # collect the selected point
        sel_pix.append(evt.index)
        # draw points
        points = []
        for idx, point in enumerate(sel_pix):
            if idx % 2 == 0:
                # draw a red circle at the handle point
                cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
            else:
                # draw a blue circle at the handle point
                cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
            points.append(tuple(point))
            # draw an arrow from handle point to target point
            if len(points) == 2:
                cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
                points = []
        return img if isinstance(img, np.ndarray) else np.array(img)

    # clear all handle/target points
    def undo_points(self,
                    original_image,
                    mask):
        if mask.sum() > 0:
            mask = np.uint8(mask > 0)
            masked_img = self.__mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
        else:
            masked_img = original_image.copy()
        return masked_img, []
    # ------------------------------------------------------

    def run_drag(self,
                seed,
                source_image, # numpy array
                mask,
                points,
                num_inference_steps,
                guidance_scale_points,
                guidance_scale_decay,
        ):

        # initialize parameters
        seed_everything(seed)

        # resize image
        source_image = Image.fromarray(source_image)
        width, height = source_image.size
        source_image = rearrange(torch.from_numpy(np.array(source_image)), 'h w c -> 1 c h w')
        source_image = 2. * source_image / 255. - 1.

        # resize mask
        mask = Image.fromarray(mask * 255)

        # trainloader: list of batch_size, where each entry is a tensor of size (N, 2)
        handle_points = []
        target_points = []
        # here, the point is in x,y coordinate
        for pi, point in enumerate(points):
            cur_point = [point[1], point[0]]
            if pi % 2 == 0:
                handle_points.append(cur_point)
            else:
                target_points.append(cur_point)
        handle_points = torch.tensor(handle_points).long()
        target_points = torch.tensor(target_points).long()

        print(handle_points)
        print(target_points)

        # output in range [0, 1]
        pred_target_image = self.pipe(
            ref_image=source_image,
            mask_image=mask,
            prompt="",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale_points=guidance_scale_points,
            guidance_scale_decay=guidance_scale_decay,
            num_guidance_steps=None,
            num_images_per_prompt=4, # set this to 4 as we are generating 4 candidate results
            output_type='pt',
            handle_points=handle_points,
            target_points=target_points,
            skip_cfg_appearance_encoder=False,
        ).images

        pred_target_image = (pred_target_image * 255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        return [Image.fromarray(pred_target_image[0]), \
            Image.fromarray(pred_target_image[1]), \
            Image.fromarray(pred_target_image[2]), \
            Image.fromarray(pred_target_image[3])]

# ------------------------------------------------------
