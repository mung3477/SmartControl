import os
import torch
from diffusers import (AutoencoderKL, ControlNetModel,
                       UniPCMultistepScheduler)
from controlnet_aux import CannyDetector, OpenposeDetector, ZoeDetector
from PIL import Image
from pytorch_lightning import seed_everything

from ip_adapter import IPAdapter
from lib import image_grid, init_store_attn_map, make_img_name, parse_args, save_attention_maps, assert_path
from smartcontrol import SmartControlPipeline, register_unet

image_dir = "./assets/images"
base_model_path = "darkstorm2150/Protogen_x3.4_Official_Release"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "models/image_encoder/"
ip_ckpt = "models/ip-adapter_sd15.bin"
preprocessor = ZoeDetector.from_pretrained("lllyasviel/Annotators")
controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
# negative_prompt_path = '/home/liuxiaoyu/compare/controlnet/realisticvision-negative-embedding'
device = "cuda"
save_attn = True

def main():
    args = parse_args()

    image_fp = f"{image_dir}/{args.ref}"
    ip_fp = f"{image_dir}/{args.ip}"
    image = Image.open(image_fp)
    control_img = preprocessor(image)
    ip_image = Image.open(ip_fp).resize((512,512))
    prompt = args.prompt
    mask_prompt = args.mask_prompt
    focus_tokens = args.focus_tokens
    pipe_options = {
        "ignore_special_tkns": True
    }

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = SmartControlPipeline.from_pretrained(
        base_model_path, controlnet=controlnet,vae=vae, torch_dtype=torch.float16
    )
    pipe.options = pipe_options
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
    init_store_attn_map(pipe)
    register_unet(
        pipe,
        smart_ckpt=None,
        mask_options={
            "alpha_mask": [1.0],
            "fixed": True
        }
    )

    seed_everything(args.seed)


    pipe(
			prompt=mask_prompt,
			image=control_img,
			# prepare_phase=True,
    ).images[0]
    save_attention_maps(
			pipe.unet.attn_maps,
			pipe.tokenizer,
			base_dir=f"{os.getcwd()}/log/attn_maps/IPAdapter/{prompt}/{mask_prompt}",
			prompts=[mask_prompt],
			options={
				"prefix": "",
				"return_dict": False,
				"ignore_special_tkns": True,
				"enabled_editing_prompts": 0
    })

    register_unet(
			pipe=pipe,
			smart_ckpt=None,
			mask_options={
				"alpha_mask": [1.0],
				"fixed": True
			},
			reset_masks=False
    )

    ip_model.pipe = pipe
    output = ip_model.generate(pil_image=ip_image, prompt=prompt, mask_prompt=mask_prompt, focus_tokens=focus_tokens, image=control_img, num_samples=1, num_inference_steps=50, controlnet_conditioning_scale = 1.0)[0]
    # output = ip_model.generate(pil_image=ip_image, prompt=prompt, image=control_img, num_samples=1, num_inference_steps=50, controlnet_conditioning_scale = 1.0)[0]

    assert_path(f"output/ip_adapter")
    output.save(f"output/ip_adapter/{prompt}.png")
    image = image_grid([ip_image.resize((256, 256)), control_img.resize((256, 256)),output.resize((256,256))], 1, 3, options={"fill": (255, 255, 255)})


    image.save(f"output/ip_adapter/{prompt} - controls.png")
    print(f"Saved at ./output/ip_adapter/{prompt}.png!")

    if save_attn:
        save_attention_maps(
        pipe.unet.attn_maps,
        pipe.tokenizer,
        base_dir=f"{os.getcwd()}/log/attn_maps/IPAdapter/{prompt}",
        prompts=[prompt],
        options={
            "prefix": "",
            "return_dict": False,
            "ignore_special_tkns": True,
            "enabled_editing_prompts": 0
        })
        save_alpha_masks(self.pipe.unet.alpha_masks, f'{os.getcwd()}/log/alpha_masks/IPAdapter/{prompt}')


if __name__ == "__main__":
    main()
