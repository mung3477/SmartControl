import argparse
from os import path

from lib import calc_diff


def prepare_attn_diff(args: argparse.Namespace):
	assert path.exists(args.diff_layer), f"{args.diff_layer} does not exists"

	attn1_path = path.join(args.diff_layer, args.attn1)
	attn2_path = path.join(args.diff_layer, args.attn2)
	attn_diff_name = f"{args.diff_layer}/diff btw {args.attn1} and {args.attn2}"

	assert path.exists(attn1_path), f"{attn1_path} does not exists"
	assert path.exists(attn2_path), f"{attn2_path} does not exists"

	args.attn1 = attn1_path
	args.attn2 = attn2_path
	args.attn_diff_name = attn_diff_name

def parse_args():
	parser = argparse.ArgumentParser(description="A brief description of your script")

	parser.add_argument("--attn_diff", action="store_true", default=False, help="Visualize difference of given attention layers")
	parser.add_argument("--diff_layer", type=str, default="./log/attn_maps/smartcontrol-a photo of tiger-depth-deer-control-1.0-alpha-[1]-multiplied-seed-12345/20/ControlNetModel.down_blocks.0.attentions.0.transformer_blocks.0.attn2/batch-0", help="Path to the attention map layer to visualize difference btw tokens")
	parser.add_argument("--attn1", type=str, default="", help="Name of the first attention map file")
	parser.add_argument("--attn2", type=str, default="", help="Name of the second attention map file")

	args = parser.parse_args()
	if args.attn_diff:
		prepare_attn_diff(args)

	return args

def main():
	args = parse_args()

	if args.attn_diff:
		calc_diff(args.attn1, args.attn2, args.attn_diff_name)




if __name__ == "__main__":
    main()
