import torch
from diffusers import (AutoencoderKL, ControlNetModel,
                       StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)
from PIL import Image
from pytorch_lightning import seed_everything

from lib import (assert_path, image_grid, init_store_attn_map, make_img_name,
                 make_ref_name, parse_args, save_alpha_masks,
                 save_attention_maps)
from smartcontrol import SmartControlPipeline, register_unet

image_dir = "./assets/images"
base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
# negative_prompt_path = '/home/liuxiaoyu/compare/controlnet/realisticvision-negative-embedding'
device = "cuda"

def use_prev_t_attn(args, control, pipe):
    image_name = make_img_name(args)
    pipe(
        prompt=args.cond_prompt,
        image=control,
        # negative_prompt=negative_prompt_path,
        controlnet_conditioning_scale = args.controlnet_conditioning_scale,
        output_name = image_name,
        prepare_phase=True
    )
    save_attention_maps(
        pipe.unet.attn_maps,
        pipe.tokenizer,
        base_dir=f"log/attn_maps/{image_name}/{args.cond_prompt}",
        prompts=[args.cond_prompt],
        options={
            "prefix": "",
            "return_dict": False,
            "ignore_special_tkns": args.ignore_special_tkns,
            "enabled_editing_prompts": args.edit_args.enabled_editing_prompts
        })

    register_unet(
        pipe,
        args.smart_ckpt,
        mask_options={
            "alpha_mask": args.alpha_mask,
            "fixed": args.alpha_fixed
        },
        reset_masks=False
    )

    output = pipe(
        prompt=args.prompt,
        mask_prompt=args.cond_prompt,
        focus_prompt = args.focus_prompt,
        image=control,
        # negative_prompt=negative_prompt_path,
        controlnet_conditioning_scale = args.controlnet_conditioning_scale,
        output_name = image_name,
        prepare_phase=False,
        edit_args=args.edit_args
    ).images[0]

    return output, image_name

def vanilla(args, control, pipe):
    image_name = make_img_name(args)
    output = pipe(
        prompt=args.prompt,
        image=control,
        # negative_prompt=negative_prompt_path,
        controlnet_conditioning_scale = args.controlnet_conditioning_scale,
        output_name = image_name,
        edit_args=args.edit_args
    ).images[0]

    return output, image_name

def main():
    args = parse_args()

    image_fp = f"{image_dir}/{args.ref}"
    image = Image.open(image_fp)
    preprocessor = args.preprocessor
    control = preprocessor(image)

    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    options = {
        "ignore_special_tkns": args.ignore_special_tkns,
        "alternate": args.alternate,
        "stop_point": args.stop_point
    }
    pipe = SmartControlPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, vae=vae, torch_dtype=torch.float16, options=options
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    init_store_attn_map(pipe)
    register_unet(
        pipe,
        args.smart_ckpt,
        mask_options={
            "alpha_mask": args.alpha_mask,
            "fixed": args.alpha_fixed
        }
    )

    seed_everything(args.seed)

    if args.alpha_attn_prev:
        output, image_name = use_prev_t_attn(args, control, pipe)
    else:
        output, image_name = vanilla(args, control, pipe)


    image = image_grid([image.resize((256, 256)), control.resize((256, 256)),output.resize((256,256))], 1, 3, caption=image_name, options={"fill": (255, 255, 255)})
    assert_path("output/vanilla/")
    image.save(f"output/vanilla/{image_name}.png")
    print(f"Saved at ./output/vanilla/{image_name}.png!")

    save_attention_maps(
        pipe.unet.attn_maps,
        pipe.tokenizer,
        base_dir=f"log/attn_maps/{image_name}",
        prompts=[args.prompt],
        options={
            "prefix": "",
            "return_dict": False,
            "ignore_special_tkns": args.ignore_special_tkns,
            "enabled_editing_prompts": args.edit_args.enabled_editing_prompts
        })
    save_alpha_masks(pipe.unet.alpha_masks, f'log/alpha_masks/{image_name}')

if __name__ == "__main__":
    main()
