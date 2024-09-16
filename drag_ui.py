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

import gradio as gr
import argparse

from utils.ui_utils import LightningDragUI

LENGTH=360 # length of the square area displaying/editing images

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--base_sd_path", type=str)
    parser.add_argument("--vae_path", type=str, default="default")
    parser.add_argument("--ip_adapter_path", type=str)
    parser.add_argument("--lightning_drag_model_path", type=str)
    parser.add_argument("--lcm_lora_path", type=str, default=None)
    parser.add_argument("--server_port", type=int, default=8888)
    args = parser.parse_args()
    return args

args = parse_args()
ui_obj = LightningDragUI(
    args.base_sd_path,
    args.vae_path,
    args.ip_adapter_path,
    args.lightning_drag_model_path,
    args.lcm_lora_path,
)

print("finished loading models")

with gr.Blocks() as demo:

    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
            LightningDrag
        </h1>
        <br>
        <h2 style="font-weight: 450; font-size: 1rem; margin: 0rem">
            <a href="https://lightning-drag.github.io/">Project page</a> | 
            <a href="https://github.com/magic-research/LightningDrag"> GitHub </a> | 
            <a href="https://arxiv.org/abs/2405.13722"> arXiv </a>
        </h2>
        </div>

        <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
            ⚠️ This demo is for academic research only. 
            Users should strictly adhere to local laws and ethics.
        </div>

        <div style="white-space: pre">
        <b>Steps (Refer to this <a href="https://github.com/magic-research/LightningDrag/blob/main/assets/demo_shortest.gif">GIF</a> for step-by-step video demo)</b>
        1. Users can either upload image or choose from examples below
        2. In the top-left window, draw mask to specify editable region
        3. In the bottom-left window, click handle and target points
        4. Press 'Run Drag' button

        <b>Precautions</b>
        1. Please ensure both handle and target points are all within masked region

        <b>FAQ</b>
        Q. What if the drag editing intention is not achieved?
        A. Please try the following options:
           1) adding more pairs of handle and target points (refer to Section 4.4.1 of our paper)
           2) change seed
           3) break down the dragging operation into a sequence of shorter dragging trajectories (refer to Section 4.4.2 of our paper)

        Q. What if the dragging results are a bit blurry and details are not very good
        A. Please try the following options:
           1) decreasing 'Guidance Scale for Points' (e.g., to 1.0 or 2.0);
           2) increasing 'Inference steps' (e.g., to 15, 25 steps).

        <b>Limitations</b>
        1. Since our model is developed on SD-1.5, it inherits some failure cases such as blurry face, distorted hands, etc.
        2. We didn't do special training to keep face identity, so it might suffer person identity loss in large magnitude head-turning editing.
        </div>
    """)

    # UI components for editing real images
    mask = gr.State(value=None) # store mask
    selected_points = gr.State([]) # store points
    original_image = gr.State(value=None) # store original input image
    with gr.Row():
        display_size = LENGTH
        with gr.Column():
            gr.Markdown("""<p style="text-align: center; font-size: 15px">Draw Mask</p>""")
            canvas = gr.Image(type="numpy", tool="sketch",
                show_label=True, height=display_size) # for mask painting
            gr.Markdown("""<p style="text-align: center; font-size: 15px">Click Points</p>""")
            input_image = gr.Image(type="numpy",
                show_label=True, height=display_size, interactive=False) # for points clicking
        with gr.Column():
            output_gallery = gr.Gallery(
                label="Dragged Images",
                show_label=True,
                elem_id="gallery",
                columns=[2],
                rows=[2],
                height=2*LENGTH,
                width=2*LENGTH,
                object_fit="contain",
                )
            with gr.Row():
                output1_button = gr.Button("Use Output1")
                output2_button = gr.Button("Use Output2")
            with gr.Row():
                output3_button = gr.Button("Use Output3")
                output4_button = gr.Button("Use Output4")
    with gr.Row():
        run_button = gr.Button("Run Drag")
        undo_button = gr.Button("Undo Points")
        clear_all_button = gr.Button("Clear All")
    with gr.Row():
        seed = gr.Number(value=42, label="Seed", precision=0)

        # Default values for LCM and non-LCM models
        default_num_inference_steps = 25 if args.lcm_lora_path is None else 8
        default_guidance_scale_points = 4.0 if args.lcm_lora_path is None else 3.0

        num_inference_steps = gr.Number(
            value=default_num_inference_steps,
            label="Inference Steps",
            precision=0)
        guidance_scale_points = gr.Number(
            value=default_guidance_scale_points,
            label="Guidance Scale for Points")
        guidance_scale_decay = gr.Textbox(
            value="inv_square",
            label="Guidance Scale Decay",
            visible=False)

    # event definition
    # event for dragging user-input real image
    canvas.edit(
        ui_obj.store_img,
        [canvas],
        [original_image, selected_points, input_image, mask]
    )
    input_image.select(
        ui_obj.get_points,
        [input_image, selected_points],
        [input_image],
    )
    undo_button.click(
        ui_obj.undo_points,
        [original_image, mask],
        [input_image, selected_points]
    )
    run_button.click(
        ui_obj.run_drag,
        [
        seed,
        original_image,
        mask,
        selected_points,
        num_inference_steps,
        guidance_scale_points,
        guidance_scale_decay,
        ],
        [output_gallery]
    )

    clear_all_button.click(
        ui_obj.clear_all,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [canvas,
        input_image,
        output_gallery,
        selected_points,
        original_image,
        mask]
    )

    output1_button.click(
        ui_obj.select_image,
        [output_gallery, 
        gr.Number(value=0, visible=False, precision=0),
        gr.Number(value=LENGTH, visible=False, precision=0),],
        [canvas,
        input_image,
        output_gallery,
        selected_points,
        original_image,
        mask]
    )
    output2_button.click(
        ui_obj.select_image,
        [output_gallery, 
        gr.Number(value=1, visible=False, precision=0),
        gr.Number(value=LENGTH, visible=False, precision=0),],
        [canvas,
        input_image,
        output_gallery,
        selected_points,
        original_image,
        mask]
    )
    output3_button.click(
        ui_obj.select_image,
        [output_gallery, 
        gr.Number(value=2, visible=False, precision=0),
        gr.Number(value=LENGTH, visible=False, precision=0),],
        [canvas,
        input_image,
        output_gallery,
        selected_points,
        original_image,
        mask]
    )
    output4_button.click(
        ui_obj.select_image,
        [output_gallery, 
        gr.Number(value=3, visible=False, precision=0),
        gr.Number(value=LENGTH, visible=False, precision=0),],
        [canvas,
        input_image,
        output_gallery,
        selected_points,
        original_image,
        mask]
    )

    # Examples
    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            "samples/pexels-h-ng-xuan-vien-1346154-2869318.jpg",
            "samples/pexels-suju-10526561.jpg",
            "samples/pexels-polina-tankilevitch-5587999.jpg",
        ],
        inputs=[canvas],
        outputs=[original_image, selected_points, input_image, mask],
        fn=ui_obj.store_img,
    )

demo.queue().launch(share=False, server_name="0.0.0.0", server_port=args.server_port)
