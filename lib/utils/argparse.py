import argparse
import warnings

from controlnet_aux import CannyDetector, OpenposeDetector, ZoeDetector
from PIL import Image

from ..experiments import _check_edit_args, _SEGA_image_name


def make_ref_name(path: str) -> str:
	return path.split("/")[-1].split(".")[0]

def make_img_name(args: argparse.Namespace) -> str:
	prompt = args.prompt
	ref_name = make_ref_name(args.ref)
	cntl_type = args.cntl
	seed = args.seed

	if args.alpha_attn_prev:
			alpha_map = "prev-timestep"
			alpha_calc = f"{args.focus_prompt} token's mask on {args.cond_prompt}"
			if args.alpha_fixed:
				alpha_calc += f"-{str(args.alpha_mask)}-on-timestep-999"
			else:
				alpha_calc += f"-SmartControl-on-timestep-999"
	elif args.alpha_fixed:
		alpha_map = str(args.alpha_mask)
		alpha_calc = "fixed"
	elif args.alpha_attn_diff:
		alpha_map = "diff"
		alpha_calc = f"{args.gen_phrase} vs {args.cond_phrase}"
		alpha_calc += f"-threshold {args.attn_diff_threshold}"
	else:
		alpha_map = str(args.alpha_mask)
		alpha_calc = "multiplied with smartcontrol"

	if args.ignore_special_tkns:
			alpha_calc += "-no <sot> <eot>"

	alternate = f"-alternate-{args.alternate}" if args.alternate else ""
	stop_point = f"-stop-at-{args.stop_point}" if args.stop_point else ""
	SEGA_suffix = _SEGA_image_name(args)

	if args.ip is None:
		return f"smartcontrol-{prompt}-{cntl_type}-{ref_name}-alpha-{alpha_map}-{alpha_calc}{alternate}{stop_point}-seed-{seed}{SEGA_suffix}"
	else:
		ip_name = make_ref_name(args.ip)
		return f"IP-smartcontrol-{prompt}-{cntl_type}-{ref_name}-IP-{ip_name}-alpha-{alpha_map}-{alpha_calc}-seed-{seed}{SEGA_suffix}"

def check_args(args: argparse.Namespace):
	if args.alpha_attn_diff is True:
		assert hasattr(args, "cond_prompt"), "You should provide condition prompt to use alpha masks inferred with cross attention differences."
		assert hasattr(args, "gen_tkn"), "You should provide a token from generation prompt to calculate cross attention difference."
		assert hasattr(args, "cond_tkn"), "You should provide a token from condition prompt to calculate cross attention difference."

		if args.gen_phrase is None:
			warnings.warn("\nYou should provide generation prompt phrase to use alpha masks inferred with cross attention differences. Set to gen_prompt.\n")
			args.gen_phrase = args.prompt

		if args.cond_phrase is None:
			warnings.warn("\nYou should provide condition prompt phrase to use alpha masks inferred with cross attention differences. Set to cond_prompt.\n")
			args.cond_phrase = args.cond_prompt


		for token in args.gen_phrase.split():
			assert token in args.prompt, f"{token} is not included in given generation prompt {args.prompt}"
		for token in args.cond_phrase.split():
			assert token in args.cond_prompt, f"{token} is not included in given condition prompt {args.cond_prompt}"

		if args.ignore_special_tkns:
			if args.gen_phrase == args.prompt:
				warnings.warn("You are ignoring <sot> and <eot>, but the given prompt and gen phrase are same. This will make the cross attention value of the phrase to 0.25\n")

			if args.cond_phrase == args.cond_prompt:
				warnings.warn("You are ignoring <sot> and <eot>, but the given cond prompt and cond phrase are same. This will make the cross attention value of the phrase to 0.25\n")

			if args.gen_phrase == args.prompt and args.cond_phrase == args.cond_prompt:
				warnings.warn("You are ignoring <sot> and <eot>, but you are using both prompts to calculate cross attention difference. This will make the difference almost ZERO\n")

	if args.alpha_attn_prev is True:
		assert args.cond_prompt is not None, "Condition prompt is needed"
		assert args.focus_prompt is not None, "Focus prompt is needed"
		for token in args.focus_prompt.split():
			assert token in args.cond_prompt, f"{token} is not included in given generation prompt {args.cond_prompt}"

		if args.alpha_fixed is False:
			warnings.warn("Current setting uses SmartControl on timestep 999.")

	_check_edit_args(args)

def decide_cntl(args: argparse.Namespace):
	if args.cntl == "depth":
		args.smart_ckpt = "./depth.ckpt"
		args.controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
		args.preprocessor = ZoeDetector.from_pretrained(args.detector_path)
	elif args.cntl == "canny":
		args.smart_ckpt = "./canny.ckpt"
		args.controlnet_path = "lllyasviel/control_v11p_sd15_canny"
		args.preprocessor = CannyDetector()
	elif args.cntl == "pose":
		args.smart_ckpt = "./depth.ckpt" 	# any ckpt is Okay since we are not using smartcontrol for pose control case
		args.controlnet_path = "lllyasviel/control_v11p_sd15_openpose"
		args.preprocessor = OpenposeDetector.from_pretrained(args.detector_path)

def parse_args():
	parser = argparse.ArgumentParser(description="A brief description of your script")

	parser.add_argument('--alpha_mask', nargs="*", type=float, default=[1], help="Mask applied on inferred alpha. [1, 0, 0, 0] means only upper left is used with 1. None uses SmartControl's inferred alpha_mask.")
	parser.add_argument('--alpha_fixed', action='store_true', default=False, help="Whether to use given alpha as fixed alpha. False means given alpha_mask is multiplied on inferred alpha elementwisely.")
	parser.add_argument('--alpha_attn_diff', action='store_true', default=False, help="Whether to calculate alpha with differences btw two cross attentions on generate prompt and condition prompt.")
	parser.add_argument('--attn_diff_threshold', type=float, default=0.0)
	parser.add_argument('--alpha_attn_prev', action='store_true', default=False, help="Whether to use previous timestep's attention map as an alpha mask.")
	parser.add_argument('--save_attn', action='store_true', default=False)
	parser.add_argument('--alternate', action='store_true', default=False, help="Alternate condition usage")
	parser.add_argument('--stop_point', type=int, default=0, help="Early stop timestep")
	parser.add_argument('--cntl', type=str, default="depth", help="Type of condition. (default: depth map)")
	parser.add_argument('--controlnet_conditioning_scale', type=float, default=1.0, help="Value of controlnet_conditioning_scale")
	parser.add_argument('--detector_path', type=str, default="lllyasviel/Annotators", help="Path to fetch pretrained control detector")
	parser.add_argument('--seed', type=int, default=12345, help="Seed")
	parser.add_argument('--prompt', type=str, required=True)
	parser.add_argument('--focus_prompt', type=str, help="Substring of given generation prompt to use corresponding cross attention map as an alpha mask")
	parser.add_argument('--cond_prompt', type=str, help="Prompt used to extract cross attention map")
	parser.add_argument('--ignore_special_tkns', action='store_true', default=False, help="Whether to ignore <sot> and <eot> while calculating cross attention differences")
	parser.add_argument('--ref', type=str, help="A path to an image that will be used as a control", required=True)
	parser.add_argument('--ip', type=str, default=None, help="A path to an image that will be used as an image prompt")

	parser.add_argument('--editing_prompt', type=str, nargs="*", default=None, help="prompts that corresponds to desired semantic latents")
	parser.add_argument('--reverse_edit_direction', type=int, nargs="*", default=None)
	parser.add_argument('--edit_warmup_steps', type=int, nargs="*", default=None)
	parser.add_argument('--edit_guidance_scale', type=float, nargs="*", default=None)
	parser.add_argument('--edit_threshold', type=float, nargs="*", default=None)
	parser.add_argument('--edit_weights', type=float, nargs="*", default=None)
	parser.add_argument('--edit_mom_scale', type=float, default=0.3)
	parser.add_argument('--edit_mom_beta', type=float, default=0.6)

	args = parser.parse_args()
	check_args(args)
	decide_cntl(args)

	return args
