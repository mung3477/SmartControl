import argparse
import os
import re
from enum import Enum

import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from lib import calc_diff, tokenize_and_mark_prompts

base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
tokenizer =AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")

class PhraseType(Enum):
    GEN = 1
    COND = 2

def parse_args():
	parser = argparse.ArgumentParser(description="A brief description of your script")

	parser.add_argument("--attn_diff", action="store_true", default=False, help="Visualize difference of given attention layers")
	parser.add_argument("--diff_layer", type=str, default="./log/attn_maps/smartcontrol-a photo of tiger-depth-deer-control-1.0-alpha-[1]-multiplied-seed-12345/20/ControlNetModel.down_blocks.0.attentions.0.transformer_blocks.0.attn2/batch-0", help="Path to the attention map layer to visualize difference btw tokens")
	parser.add_argument("--gen_phrase", type=str, default="", help="Name of the first attention map files")
	parser.add_argument("--cond_phrase", type=str, default="", help="Name of the second attention map files")
	parser.add_argument('--ignore_special_tkns', action='store_true', default=False, help="Whether to ignore <sot> and <eot> while calculating cross attention differences")

	args = parser.parse_args()
	if args.attn_diff:
		prepare_attn_diff(args)

	return args

def prepare_attn_diff(args: argparse.Namespace):
	assert os.path.exists(args.diff_layer), f"{args.diff_layer} does not exists"

	attn_diff_name = f"{args.diff_layer}/diff btw {args.cond_phrase} and {args.gen_phrase}"
	args.attn_diff_name = attn_diff_name

def aggregate(dir: str, phrase: str, type: PhraseType, ignore_special_tkns: bool):
	tokens = tokenize_and_mark_prompts([phrase], tokenizer, ignore_special_tokens=ignore_special_tkns)[0]
	files = []
	for token in tokens:
		pattern = r"\d+" + re.escape(f"-{token}.png")
		if type == PhraseType.COND:
			pattern = r"sub\-" + pattern

		matched = list(filter(lambda s: re.match(pattern, s), os.listdir(dir)))
		assert len(matched) == 1, f"{dir} has {len(matched)} files whose name includes {token}"

		files.append(matched[0])

	files = [f"{dir}/{file}" for file in files]

	aggregated = None
	for file in files:
		attn = np.array(Image.open(file))
		if aggregated is None:
			aggregated = attn
		else:
			aggregated += attn
	return aggregated // len(files)

def main():
	args = parse_args()
	ignore_special_tkns = args.ignore_special_tkns
	gen_attn = aggregate(args.diff_layer, args.gen_phrase, PhraseType.GEN, ignore_special_tkns)
	gen_attn = Image.fromarray(gen_attn)
	cond_attn = aggregate(args.diff_layer, args.cond_phrase, PhraseType.COND, ignore_special_tkns)
	cond_attn = Image.fromarray(cond_attn)

	name = f"{args.attn_diff_name}"
	if ignore_special_tkns:
		name += " special token ignored"

	if args.attn_diff:
		calc_diff(cond_attn, gen_attn, name)


if __name__ == "__main__":
    main()
