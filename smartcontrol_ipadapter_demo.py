import numpy as np
import torch
from controlnet_aux import ZoeDetector
from diffusers import (AutoencoderKL, ControlNetModel,
                       StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)
from PIL import Image
from pytorch_lightning import seed_everything

from ip_adapter import IPAdapter
from lib import image_grid

prompt = 'a monster toy doing bicycle kick'
ip_fp = "assets/images/monster_toy.png"
ip_name = ip_fp.split("/")[-1].split(".")[0]


base_model_path = "darkstorm2150/Protogen_x3.4_Official_Release"
vae_model_path = "stabilityai/sd-vae-ft-mse"
controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
depth_path = 'lllyasviel/Annotators'
smart_ckpt = "./depth.ckpt"
# negative_prompt_path = '/home/liuxiaoyu/compare/controlnet/realisticvision-negative-embedding'

image_encoder_path = "models/image_encoder/"
ip_ckpt = "models/ip-adapter_sd15.bin"

device = "cuda"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet,vae=vae, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()


preprocessor = ZoeDetector.from_pretrained(depth_path)
image = Image.open("assets/images/overhead_kick.png")
depth_map = preprocessor(image)
ip_image = Image.open(ip_fp)
depth_map = depth_map.resize((512,512))
ip_image = ip_image.resize((512,512))

from lib import init_store_attn_map
from smartcontrol import register_unet

# pipe = init_store_attn_map(pipe)
pipe = register_unet(pipe,smart_ckpt)
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

seed_everything(12345)

print(np.array(ip_image).shape)
output = ip_model.generate(pil_image=ip_image, image=depth_map, num_samples=1, num_inference_steps=50, controlnet_conditioning_scale = 0.8)[0]
image = image_grid([ip_image.resize((256, 256)), depth_map.resize((256, 256)), output.resize((256, 256))], 1, 3, prompt)
image.save(f"output/{prompt}-{ip_name}-IP-smartcontrol.png")
