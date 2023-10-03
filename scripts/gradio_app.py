import random
import argparse
from functools import partial
import numpy as np
import gradio as gr
from omegaconf import OmegaConf
import torch 

from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1):
    if type(prompt)!=list:
        prompt = [prompt]
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size,1,1)}
        uc_ = {"context": uc.repeat(batch_size,1,1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(S=step, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_,
                                        eta=ddim_eta, x_T=None)
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


def generate_images(args, model, sampler, text_input, uncond_text_input, seed, guidance_scale, step, elevation, azimuth, use_camera):
    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    batch_size = args.num_frames

    if use_camera:
        camera = get_camera(args.num_frames, elevation=elevation, azimuth_start=azimuth)
        camera = camera.repeat(batch_size//args.num_frames,1).to(device)
        num_frames = args.num_frames
    else:
        camera = None
        num_frames = 1
    
    t = text_input + args.suffix
    uc = model.get_learned_conditioning( [uncond_text_input] ).to(device)
    set_seed(seed)
    images = []
    for _ in range(2):
        img = t2i(model, args.size, t, uc, sampler, step=step, scale=guidance_scale, batch_size=batch_size, ddim_eta=0.0, 
                dtype=dtype, device=device, camera=camera, num_frames=num_frames)
        img = np.concatenate(img, 1)
        images.append(img)
    images = np.concatenate(images, 0)
    return images



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default=None, help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to local checkpoint")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    print("load t2i model ... ")
    if args.config_path is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt_path)
    else:
        assert args.ckpt_path is not None, "ckpt_path must be specified!"
        config = OmegaConf.load(args.config_path)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    model.device = args.device
    model.to(args.device)
    model.eval()

    sampler = DDIMSampler(model)
    print("load t2i model done . ")

    fn_with_model = partial(generate_images, args, model, sampler)

    with gr.Blocks() as demo:
        gr.Markdown("MVDream demo for multi-view images generation from text and camera inputs.")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(value="", label="prompt")
                uncond_text_input = gr.Textbox(value="", label="negative prompt")
                seed = gr.Number(value=23, label="seed", precision=0)
                guidance_scale = gr.Number(value=7.5, label="guidance_scale")
                step = gr.Number(value=25, label="sample steps", precision=0)
                elevation = gr.Slider(0, 30, value=15, label="Elevation", info="Choose between 0 and 30")
                azimuth = gr.Slider(0, 360, value=0, label="Azimuth", info="Choose between 0 and 360")
                use_camera = gr.Checkbox(value=True, label="Multi-view Mode", info="Multi-view mode or not (indepedent images)")
                text_button = gr.Button("Generate Images")
            with gr.Column():
                image_output = gr.Image()

        inputs = [text_input, uncond_text_input, seed, guidance_scale, step, elevation, azimuth, use_camera]
        default_params = ["", 23, 7.5, 30, 15, 0, True]
        gr.Examples(
            [   
                ["an astronaut riding a horse"] + default_params,
                ["an earth"] + default_params,
                ["a statue of a cute cat"] + default_params,
                ["Luffy from one piece, head, super detailed, best quality, 4K, HD"] + default_params,
                ["higly detailed, majestic royal tall ship, realistic painting, by Charles Gregory Artstation and Antonio Jacobsen and Edward Moran, intricated details, blender, hyperrealistic, 4k, HD"] + default_params,
            ],
            inputs,
            image_output,
            fn_with_model,
            cache_examples=True,
        )

        text_button.click(fn_with_model, inputs=inputs, outputs=image_output)

    demo.launch(share=True)
