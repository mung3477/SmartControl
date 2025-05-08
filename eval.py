import argparse
import os
from test.evaluate import QuantitativeEval
from test.types import ConflictDegree, ModelType


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0")
	parser.add_argument("--conflict_degree", type=str, default="no_conflict", help="no_conflict | mild_conflict | significant_conflict")
	parser.add_argument("--model_type", type=str, default="ControlAttend", help="ControlNet | SmartControl | ControlAttend")
	parser.add_argument("--attn_bias", type=float, default=0.0, help="0.0 | 1.0 | 3.0")
	parser.add_argument("--controlnet_alpha", type=float, default=1.0, help="0.4 | 1.0")

	args = parser.parse_args()
	args.conflict_degree = ConflictDegree.str2enum(args.conflict_degree)
	args.modelType = ModelType.str2enum(args.model_type)

	return args

def _get_log_name(args):
	log_name=f"{args.conflict_degree.name}-{args.modelType.name}"

	if args.modelType.name == "ControlNet":
		log_name += f"-alpha-{args.controlnet_alpha}"

	if args.modelType.name == "ControlAttend":
		log_name += f"-bias-{args.attn_bias}"

	return log_name

def main():
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

	QE = QuantitativeEval()
	images_info = QE.get_all_image_pathes(conflict_degree=args.conflict_degree, modelType=args.modelType, bias=args.attn_bias, controlnet_alpha=args.controlnet_alpha)
	log_name = _get_log_name(args)

	QE.evaluate_results(
		log_name=log_name,
		image_ref_prompt_pairs=images_info,
		self_simil=True,
		img_reward=True,
		clip=True
    )
	# QE.record_top_image_reward_file("/root/Desktop/workspace/Daewon/SmartControl/test/human_eval/significant")

if __name__ == "__main__":
	main()
