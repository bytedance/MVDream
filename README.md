# MVDream
Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, Xiao Yang

| [Project Page](https://mv-dream.github.io/) | [3D Generation](https://github.com/bytedance/MVDream-threestudio) | [Paper](https://arxiv.org/abs/2308.16512) | [HuggingFace Demo (Coming)]() |

![multiview diffusion](https://github.com/bytedance/MVDream/assets/21265012/849cb798-1d97-42fd-9f02-c23b0dc507d5)

## 3D Generation

- **This repository only includes the diffusion model and 2D image generation code of [MVDream](https://mv-dream.github.io/index.html) paper.**
- **For 3D Generation, please check [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio).**


## Installation
You can use the same environment as in [Stable-Diffusion](https://github.com/Stability-AI/stablediffusion) for this repo. Or you can set up the environment by installing the given requirements
``` bash
pip install -r requirements.txt
```

To use MVDream as a python module, you can install it by `pip install -e .` or:
``` python
pip install git+https://github.com/bytedance/MVDream
```

## Model Card
Our models are provided on the [Huggingface Model Page](https://huggingface.co/MVDream/MVDream/) with the OpenRAIL license.
| Model      | Base Model | Resolution |
| ----------- | ----------- | ----------- |
| sd-v2.1-base-4view   | [Stable Diffusion 2.1 Base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) | 4x256x256 |
| sd-v1.5-4view        | [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)             | 4x256x256 |

By default, we use the SD-2.1-base model in our experiments. 

Note that you don't have to manually download the checkpoints for the following scripts.


## Text-to-Image

You can simply generate multi-view images by running the following command:

``` bash
python scripts/t2i.py --text "an astronaut riding a horse"
```
We also provide a gradio script to try out with GUI:

``` bash
python scripts/gradio_app.py
```

## Usage
#### Load the Model
We provide two ways to load the models of MVDream:
- **Automatic**: load the model config with model name and weights from huggingface.
``` python
from mvdream.model_zoo import build_model
model = build_model("sd-v2.1-base-4view")
```
- **Manual**: load the model with a config file and a checkpoint file.
``` python
from omegaconf import OmegaConf
from mvdream.ldm.util import instantiate_from_config
config = OmegaConf.load("mvdream/configs/sd-v2-base.yaml")
model = instantiate_from_config(config.model)
model.load_state_dict(torch.load("path/to/sd-v2.1-base-4view.th", map_location='cpu'))
```

#### Inference
Here is a simple example for model inference:
``` python
import torch
from mvdream.camera_utils import get_camera
model.eval()
model.cuda()
with torch.no_grad():
    noise = torch.randn(4,4,32,32, device="cuda") # batch of 4x for 4 views, latent size 32=256/8
    t = torch.tensor([999]*4, dtype=torch.long, device="cuda") # same timestep for 4 views
    cond = {
        "context": model.get_learned_conditioning([""]*4).cuda(), # text embeddings
        "camera": get_camera(4).cuda(),
        "num_frames": 4,
    }
    eps = model.apply_model(noise, t, cond=cond)
```


## Acknowledgement
This repository is heavily based on [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1-base). We would like to thank the authors of these work for publicly releasing their code.

## Citation
``` bibtex
@article{shi2023MVDream,
  author = {Shi, Yichun and Wang, Peng and Ye, Jianglong and Mai, Long and Li, Kejie and Yang, Xiao},
  title = {MVDream: Multi-view Diffusion for 3D Generation},
  journal = {arXiv:2308.16512},
  year = {2023},
}
```
