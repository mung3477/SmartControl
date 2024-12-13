import argparse
import warnings

from controlnet_aux import CannyDetector, ZoeDetector
from PIL import Image


def make_ref_name(path: str) -> str:
    return path.split("/")[-1].split(".")[0]

def make_img_name(args: argparse.Namespace) -> str:
	prompt = args.prompt
	ignore_special_tkns: args.ignore_special_tkns
	ref_name = make_ref_name(args.ref)
	cntl_type = args.cntl
	controlnet_conditioning_scale = args.controlnet_conditioning_scale
	seed = args.seed

	if args.alpha_fixed:
		alpha_map = str(args.alpha_mask)
		alpha_calc = "fixed"
	elif args.alpha_attn_diff:
		alpha_map = "diff"
		alpha_calc = f"{' '.join(args.gen_phrase)} vs {' '.join(args.cond_phrase)}"
		if args.ignore_special_tkns:
			alpha_calc += "-no <sot> <eot>"
	else:
		alpha_map = str(args.alpha_mask)
		alpha_calc = "multiplied with smartcontrol"


	if args.ip is None:
		return f"smartcontrol-{prompt}-{cntl_type}-{ref_name}-control-{controlnet_conditioning_scale}-alpha-{alpha_map}-{alpha_calc}-seed-{seed}"
	else:
		ip_name = make_ref_name(args.ip)
		return f"IP-smartcontrol-{prompt}-{cntl_type}-{ref_name}-IP-{ip_name}-control-{controlnet_conditioning_scale}-alpha-{alpha_map}-{alpha_calc}-seed-{seed}"

def check_args(args: argparse.Namespace):
	if args.alpha_attn_diff is True:
		assert hasattr(args, "cond_prompt"), "You should provide condition prompt to use alpha masks inferred with cross attention differences."
		assert hasattr(args, "cond_phrase"), "You should provide condition prompt phrase to use alpha masks inferred with cross attention differences."

		for token in args.gen_phrase:
			assert token in args.prompt, f"{token} is not included in given generation prompt {args.prompt}"
		for token in args.cond_phrase:
			assert token in args.cond_prompt, f"{token} is not included in given condition prompt {args.cond_prompt}"

		if args.ignore_special_tkns:
			if args.gen_phrase == args.prompt:
				warnings.warn("You are ignoring <sot> and <eot>, but the given prompt and gen phrase are same. This will make the cross attention value of the phrase to 0.25\n")

			if args.cond_phrase == args.cond_prompt:
				warnings.warn("You are ignoring <sot> and <eot>, but the given cond prompt and cond phrase are same. This will make the cross attention value of the phrase to 0.25\n")

			if args.gen_phrase == args.prompt and args.cond_phrase == args.cond_prompt:
				warnings.warn("You are ignoring <sot> and <eot>, but you are using both prompts to calculate cross attention difference. This will make the difference almost ZERO\n")

		args.gen_phrase = args.gen_phrase.split()
		args.cond_phrase = args.cond_phrase.split()


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
	parser.add_argument('--alpha_attn_diff', action='store_true', default=False, help="Whether to calculate alpha with differences btw two cross attentions on generate prompt and condition prompt.")
	parser.add_argument('--cntl', type=str, default="depth", help="Type of condition. (default: depth map)")
	parser.add_argument('--controlnet_conditioning_scale', type=float, default=1.0, help="Value of controlnet_conditioning_scale")
	parser.add_argument('--detector_path', type=str, default="lllyasviel/Annotators", help="Path to fetch pretrained control detector")
	parser.add_argument('--seed', type=int, default=12345, help="Seed")
	parser.add_argument('--prompt', type=str, required=True)
	parser.add_argument('--gen_phrase', type=str, required=True, help="Substring of given generation prompt to calculate cross attention differences")
	parser.add_argument('--cond_prompt', type=str, help="Prompt to be cross-attentioned with condition image latent")
	parser.add_argument('--cond_phrase', type=str, help="Substring of given condition prompt to calculate cross attention differences")
	parser.add_argument('--ignore_special_tkns', action='store_true', default=False, help="Whether to ignore <sot> and <eot> while calculating cross attention differences")
	parser.add_argument('--ref', type=str, help="A path to an image that will be used as a control", required=True)
	parser.add_argument('--ip', type=str, default=None, help="A path to an image that will be used as an image prompt")

	args = parser.parse_args()
	check_args(args)
	decide_cntl(args)

	return args
