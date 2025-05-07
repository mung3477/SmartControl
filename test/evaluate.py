import csv
import os
import shutil
from typing import List, Optional, Tuple

import clip
import ImageReward as RM
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize
from tqdm import tqdm

from .evaluate_vit import VitExtractor
from .types import ConflictDegree, ModelType


class QuantitativeEval():
	@staticmethod
	def _self_similarity_score(device):
		""""
			########### self_similarity ###########
			self_similarity_fn = self_similarity_score(device)

			image_path1 = 'assets/images/Cheer.jpg'
			image_path2 = 'assets/images/Gesture.jpg'

			score = self_similarity_fn(image_path1, image_path2)
			print(f"Self Similarity Score: {score}")
		"""
		model_name = 'dino_vitb8'
		# dino_global_patch_size = 224

		extractor = VitExtractor(model_name=model_name, device=device)
		imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		global_resize_transform = Resize((256,256))
		global_transform = transforms.Compose([transforms.ToTensor(), global_resize_transform, imagenet_norm])

		def _fn(source_image_path:str, target_image_path:str):
			source_image = Image.open(source_image_path).convert('RGB')
			target_image = Image.open(target_image_path).convert('RGB')

			source_image = global_transform(source_image).unsqueeze(0).to(device)
			target_image = global_transform(target_image).unsqueeze(0).to(device)

			with torch.no_grad():
				source_sim = extractor.get_keys_self_sim_from_input(source_image, layer_num=11)
				target_sim = extractor.get_keys_self_sim_from_input(target_image, layer_num=11)

			score = torch.nn.functional.mse_loss(source_sim, target_sim).item()
			return score

		return _fn

	@staticmethod
	def _image_reward_score(device):
		"""
			########### image_reward ###########
			image_reward_fn = image_reward_score(device)

			image_path = 'assets/images/Guitar.png'
			prompt = "a man playing a guitar"

			score = image_reward_fn(image_path, prompt)
			print(f"Image Reward Score: {score}")
		"""
		model = RM.load("ImageReward-v1.0").to(device)

		def _fn(image_path:str, prompt:str):
			return model.score(prompt, [image_path])

		return _fn

	@staticmethod
	def _clip_score(device):
		"""
			########### clip ###########
			clip_fn = clip_score(device)

			image_path = 'assets/images/Guitar.png'
			prompt = "a man playing a guitar"

			score = clip_fn(image_path, prompt)
			print(f"CLIP Score: {score}")
		"""
		model, preprocess = clip.load('ViT-B/32')
		model.eval()
		model.to(device)
		cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

		def _fn(image_path:str, prompt:str):
			img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
			text = clip.tokenize([prompt]).to(device)

			with torch.no_grad():
				image_feature = model.encode_image(img)
				text_embedding = model.encode_text(text)
				score = cos_sim(text_embedding, image_feature).item()
			return score

		return _fn

	@staticmethod
	def _parse_filepath(fp: str):
		parts = fp.split('/')
		conflict_degree = parts[-6]  # mild
		subject = parts[-5]         # subject
		prompt_subject = parts[-4]  # prompt_subject
		prompt = parts[-3]         # prompt
		model = parts[-2]          # model
		control = parts[-1]        # control

		return (conflict_degree, subject, prompt_subject, prompt, model, control)

	@staticmethod
	def _get_output_root(conflict_degree: ConflictDegree):
		conflict = ConflictDegree.mild_conflict.name \
			if conflict_degree == ConflictDegree.no_conflict \
			else conflict_degree.name

		return f"{os.getcwd()}/test/human_eval/{conflict}"

	@staticmethod
	def _get_cntl_img(conflict_degree: ConflictDegree, subject, prompt_subject, prompt):
		conflict = ConflictDegree.mild_conflict.name \
			if conflict_degree == ConflictDegree.no_conflict \
			else conflict_degree.name
		root_dir = f"{os.getcwd()}/assets/test/selected/{conflict}"
		filename = f"{prompt.replace(prompt_subject, subject)}"
		return f"{root_dir}/{filename}.jpg"

	@staticmethod
	def _is_same_conflict_degree(conflict_degree: ConflictDegree, subject: str, prompt_subject: str):
		if conflict_degree == ConflictDegree.no_conflict:
			return subject == prompt_subject
		elif conflict_degree == ConflictDegree.mild_conflict:
			return subject != prompt_subject
		return True


	@staticmethod
	def get_all_image_pathes(conflict_degree: ConflictDegree, modelType: ModelType, bias: float, controlnet_alpha: float):
		model_name = modelType.name
		if bias != 0.0:
			model_name = f"{model_name}-bias-{bias}"
		output_root = QuantitativeEval._get_output_root(conflict_degree)
		assert os.path.exists, f"{output_root} does not exist"

		all_image_pathes = []   # (generated, refernce, prompt)
		for root, _, files in os.walk(output_root):
			for file in files:
				if file.endswith('.png') and ("seed" in file) and ("result" not in file) and file[file.find("seed") + 5:-4].isdigit():
					(_, subject, prompt_subject, prompt, model, control) \
						= QuantitativeEval._parse_filepath(root)

					if model == model_name and control == "depth" and \
						(model_name != "ControlNet" or str(controlnet_alpha) in file) and \
						QuantitativeEval._is_same_conflict_degree(conflict_degree, subject, prompt_subject):
							cntl_img = QuantitativeEval._get_cntl_img(conflict_degree, subject, prompt_subject, prompt)
							all_image_pathes.append((os.path.join(root, file), cntl_img, prompt))
		return all_image_pathes

	@staticmethod
	def get_all_sample_dirs(root: str):
		leaf_dirs = []
		for dirpath, dirnames, filenames in os.walk(root):
			if not dirnames:  # Check if it's a leaf directory (no subdirectories)
				leaf_dirs.append(dirpath)
		return leaf_dirs

	@staticmethod
	def _record_as_csv(log_name: str, self_simil_score: Optional[float], img_reward_score: Optional[float], clip_score: Optional[float]):
		csv_file_path = f"{os.getcwd()}/test/evaluation_scores.csv"
		csv_columns = ["log_name", "self_similarity", "image_reward", "clip"]
		try:
			with open(csv_file_path, 'a', newline='') as csvfile:
				writer = csv.writer(csvfile, delimiter='\t')
				writer.writerow([log_name, str(self_simil_score), str(img_reward_score), str(clip_score)])
		except Exception as e:
			print(f"Error writing to CSV: {e}")

	def __init__(self, device = "cuda"):
		self.measure_self_sim = self._self_similarity_score(device)
		self.measure_img_rwd = self._image_reward_score(device)
		self.measure_clip_scr = self._clip_score(device)

	def evaluate_results(self, log_name: str, image_ref_prompt_pairs: List[Tuple[str, str, str]], self_simil: bool = True, img_reward: bool = True, clip: bool = True):
		self_simil_score = 0.0 if self_simil is True else None
		img_reward_score =  0.0 if img_reward is True else None
		clip_score =  0.0 if clip is True else None

		csv_file_path = f"{os.getcwd()}/test/self_sim_scores/self_sim_{log_name}.csv"
		with open(csv_file_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter='\t')
			writer.writerow(["score", "name"])

		for img_fp, ref_fp, prompt in tqdm(image_ref_prompt_pairs, desc="Evaluating images"):
			if self_simil_score is not None:
				score = self.measure_self_sim(img_fp, ref_fp)

				with open(csv_file_path, 'a', newline='') as csvfile:
					writer = csv.writer(csvfile, delimiter='\t')
					writer.writerow([str(score), "_".join(img_fp.split("/")[-2:])])

				self_simil_score += score
			if img_reward_score is not None:
				img_reward_score += self.measure_img_rwd(img_fp, prompt)
			if clip_score is not None:
				clip_score += self.measure_clip_scr(img_fp, prompt)

		if self_simil_score is not None:
			self_simil_score /= len(image_ref_prompt_pairs)
			self_simil_score = round(self_simil_score, 4)
		if img_reward_score is not None:
			img_reward_score /= len(image_ref_prompt_pairs)
			img_reward_score = round(img_reward_score, 4)
		if clip_score is not None:
			clip_score /= len(image_ref_prompt_pairs)
			clip_score = round(clip_score, 4)

		print(f"######### {log_name} ########")
		if self_simil_score is not None:
			print(f"self similarity score: {self_simil_score}")
		if img_reward_score is not None:
			print(f"image reward score: {img_reward_score}")
		if clip_score is not None:
			print(f"clip score: {clip_score}")

		self._record_as_csv(log_name, self_simil_score, img_reward_score, clip_score)

	def record_top_image_reward_file(self, root: str):
		dirs = self.get_all_sample_dirs(root)
		for dir in dirs:
			image_reward_scores = []
			files = [file for file in os.listdir(dir) if file.endswith(".png") and "result" not in file and "seed" in file]

			for file in files:
				prompt = dir.split("/")[-2]
				image_reward_scores.append(self.measure_img_rwd(os.path.join(dir, file), prompt))
			max_index = image_reward_scores.index(max(image_reward_scores))
			shutil.copy(os.path.join(dir, files[max_index]), os.path.join(dir, f"top_image_reward.png"))
			print(f"Marked {os.path.join(dir, files[max_index])} as the top image reward score image")

