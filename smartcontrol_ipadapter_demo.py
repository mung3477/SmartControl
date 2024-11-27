import torch
from diffusers import (AutoencoderKL, ControlNetModel,
                       StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)
from PIL import Image
from pytorch_lightning import seed_everything

from ip_adapter import IPAdapter
from lib import image_grid, init_store_attn_map, make_img_name, parse_args
from smartcontrol import register_unet

image_dir = "./assets/images"
base_model_path = "darkstorm2150/Protogen_x3.4_Official_Release"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "models/image_encoder/"
ip_ckpt = "models/ip-adapter_sd15.bin"
# negative_prompt_path = '/home/liuxiaoyu/compare/controlnet/realisticvision-negative-embedding'
device = "cuda"

def main():
    args = parse_args()

    image_fp = f"{image_dir}/{args.ref}"
    ip_fp = f"{image_dir}/{args.ip}"
    image = Image.open(image_fp)
    preprocessor = args.preprocessor
    control = preprocessor(image)
    ip_image = Image.open(ip_fp).resize((512,512))
    prompt = args.prompt

    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet,vae=vae, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe = register_unet(
        pipe,
        args.smart_ckpt,
        mask_options={
            "alpha_mask": args.alpha_mask,
            "fixed": args.alpha_fixed
        }
    )
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

    seed_everything(args.seed)
    output = ip_model.generate(pil_image=ip_image, prompt=prompt, image=control, num_samples=1, num_inference_steps=50, controlnet_conditioning_scale = 0.8)[0]

    image_name = make_img_name(args)
    image = image_grid([ip_image.resize((256, 256)), control.resize((256, 256)),output.resize((256,256))], 1, 3, caption=image_name, options={"fill": (255, 255, 255)})
    image.save(f"output/ip_adapter/{image_name}.png")
    print(f"Saved at ./output/ip_adapter/{image_name}.png!")

if __name__ == "__main__":
    main()
