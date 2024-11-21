import torch
from controlnet_aux import ZoeDetector
from diffusers import (AutoencoderKL, ControlNetModel,
                       StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)
from PIL import Image
from pytorch_lightning import seed_everything

from lib import image_grid, init_store_attn_map, save_attention_maps
from smartcontrol import register_unet

prompt = 'a girl doing a bicycle kick'
ref_fp = "assets/images/overhead_kick.png"
ref_name = ref_fp.split("/")[-1].split(".")[0]

base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
depth_path = 'lllyasviel/Annotators'
smart_ckpt = "./depth.ckpt"
# negative_prompt_path = '/home/liuxiaoyu/compare/controlnet/realisticvision-negative-embedding'
device = "cuda"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
preprocessor = ZoeDetector.from_pretrained(depth_path)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet,vae=vae, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

image = Image.open(ref_fp)
depth_map = preprocessor(image)

pipe = init_store_attn_map(pipe)
pipe = register_unet(pipe,smart_ckpt)

seed_everything(42)
output = pipe(
    prompt=prompt,
    image=depth_map,
    # negative_prompt=negative_prompt_path,
    controlnet_conditioning_scale = 0.25

).images[0]

image = image_grid([image.resize((256, 256)), depth_map.resize((256, 256)),output.resize((256,256))], 1, 3, prompt, options={"fill": (255, 255, 255)})
image.save(f"output/{prompt}-{ref_name}-smartcontrol.png")

save_attention_maps(pipe.unet.attn_maps, pipe.tokenizer, prompts=[prompt])
