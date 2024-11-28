import argparse

from controlnet_aux import CannyDetector, ZoeDetector
from PIL import Image


def make_ref_name(path: str) -> str:
    return path.split("/")[-1].split(".")[0]

def make_img_name(args: argparse.Namespace) -> str:
	prompt = args.prompt
	ref_name = make_ref_name(args.ref)
	cntl_type = args.cntl
	controlnet_conditioning_scale = args.controlnet_conditioning_scale
	alpha_map = "inferred" if args.alpha_mask is None else str(args.alpha_mask)
	fixed = "fixed" if args.alpha_fixed is True else "multiplied"
	seed = args.seed

	if args.ip is None:
		return f"smartcontrol-{prompt}-{cntl_type}-{ref_name}-control-{controlnet_conditioning_scale}-alpha-{alpha_map}-{fixed}-seed-{seed}"
	else:
		ip_name = make_ref_name(args.ip)
		return f"IP-smartcontrol-{prompt}-{cntl_type}-{ref_name}-IP-{ip_name}-control-{controlnet_conditioning_scale}-alpha-{alpha_map}-{fixed}-seed-{seed}"

	return name

def decide_cntl(args: argparse.Namespace):
    if args.cntl == "depth":
        args.smart_ckpt = "./depth.ckpt"
        args.controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
        args.preprocessor = ZoeDetector.from_pretrained(args.detector_path)
    elif args.cntl == "canny":
        args.smart_ckpt = "./canny.ckpt"
        args.controlnet_path = "lllyasviel/control_v11p_sd15_canny"
        args.preprocessor = CannyDetector()

def parse_args():
	parser = argparse.ArgumentParser(description="A brief description of your script")

	parser.add_argument('--alpha_mask', nargs="*", type=float, default=[1], help="Mask applied on inferred alpha. [1, 0, 0, 0] means only upper left is used with 1. None uses SmartControl's inferred alpha_mask.")
	parser.add_argument('--alpha_fixed', action='store_true', default=False, help="Whether to use given alpha as fixed alpha. False means given alpha_mask is multiplied on inferred alpha elementwisely.")
	parser.add_argument('--cntl', type=str, default="depth", help="Type of condition. (default: depth map)")
	parser.add_argument('--controlnet_conditioning_scale', type=float, default=1.0, help="Value of controlnet_conditioning_scale")
	parser.add_argument('--detector_path', type=str, default="lllyasviel/Annotators", help="Path to fetch pretrained control detector")
	parser.add_argument('--seed', type=int, default=12345, help="Seed")
	parser.add_argument('--prompt', type=str, required=True)
	parser.add_argument('--cond_prompt', type=str, help="Prompt to be cross-attentioned with condition image latent")
	parser.add_argument('--ref', type=str, help="A path to an image that will be used as a control", required=True)
	parser.add_argument('--ip', type=str, default=None, help="A path to an image that will be used as an image prompt")

	args = parser.parse_args()
	decide_cntl(args)

	return args
