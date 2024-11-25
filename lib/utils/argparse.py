import argparse
from typing import List

from controlnet_aux import CannyDetector, ZoeDetector


def make_ref_name(path: str) -> str:
    return path.split("/")[-1].split(".")[0]

def make_img_name(args: argparse.Namespace) -> str:
	prompt = args.prompt
	ref_name = make_ref_name(args.ref)
	cntl_type = args.cntl
	controlnet_conditioning_scale = args.controlnet_conditioning_scale
	seed = args.seed
	name = f"output/smartcontrol-{prompt}-{cntl_type}-{ref_name}-control-{controlnet_conditioning_scale}-seed-{seed}.png"

	return name

def decide_cntl(args: argparse.Namespace):
    if args.cntl is "depth":
        args.smart_ckpt = "./depth.ckpt"
        args.controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
        args.preprocessor = ZoeDetector.from_pretrained(args.detector_path)
    elif args.cntl is "canny":
        args.smart_ckpt = "./canny.ckpt"
        args.controlnet_path = "lllyasviel/control_v11f1p_sd15_canny"
        args.preprocessor = CannyDetector.from_pretrained(args.detector_path)

def parse_args():
	parser = argparse.ArgumentParser(description="A brief description of your script")

	parser.add_argument('--alpha_map', nargs="*", type=int, default=None, help="Fixed alpha map. [1, 0, 0, 0] means only upper left is used with 1. None uses SmartControl's inferred alpha map.")
	parser.add_argument('--cntl', type=str, default="depth", help="Type of condition. (default: depth map)")
	parser.add_argument('--controlnet_conditioning_scale', type=float, default=1.0, help="Value of controlnet_conditioning_scale")
	parser.add_argument('--detector_path', type=str, default="lllyasviel/Annotators", help="Path to fetch pretrained control detector")
	parser.add_argument('--seed', type=int, default=12345, help="Seed")
	parser.add_argument('--prompt', type=str, required=True)
	parser.add_argument('--ref', type=str, help="A path to an image that will be used as a control", required=True)

	args = parser.parse_args()
	decide_cntl(args)

	return args
