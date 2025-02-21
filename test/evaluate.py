import os
from typing import Tuple, List, Optional
import csv

import clip
# import ImageReward as RM
import torch
from tqdm import tqdm

from .evaluate_vit import VitExtractor
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize

from .types import ModelType, ConflictDegree

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
	def _get_promptNref_fp(dir_path: str):
		[prompt, ref_fp] = dir_path.split("/")[-1].split(" with ")
		prompt = prompt.split(" - ")[-1]

		subject = " ".join(ref_fp.split(' ')[:2])
		action = " ".join(ref_fp.split(' ')[2:])
		ref_fp = f"/root/Desktop/workspace/SmartControl/assets/test/{subject}/{action}.png"

		return (prompt, ref_fp)

	@staticmethod
	def get_all_image_pathes(conflict_degree: ConflictDegree, modelType: ModelType):
		output_root = f"/root/Desktop/workspace/SmartControl/test/output/{conflict_degree.name}/{modelType.name}"
		assert os.path.exists, f"{output_root} does not exist"

		all_image_pathes = []   # (generated, refernce, prompt)
		for root, _, files in os.walk(output_root):
			for file in files:
				if file.endswith(('.png')) and "generated" in file:
					prompt, ref_fp = QuantitativeEval._get_promptNref_fp(root)
					all_image_pathes.append((os.path.join(root, file), ref_fp, prompt))
		return all_image_pathes

	@staticmethod
	def _record_as_csv(log_name: str, self_simil_score: Optional[float], img_reward_score: Optional[float], clip_score: Optional[float]):
		csv_file_path = "/root/Desktop/workspace/SmartControl/test/evaluation_scores.csv"
		csv_columns = ["log_name", "self_similarity", "image_reward", "clip"]
		try:
			with open(csv_file_path, 'a', newline='') as csvfile:
				writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
				writer.writerow({
					"log_name": log_name,
					"self_similarity": str(self_simil_score),
					"image_reward": str(img_reward_score),
					"clip": str(clip_score)
				})
		except Exception as e:
			print(f"Error writing to CSV: {e}")

	def __init__(self, device = "cuda"):
		# self.measure_self_sim = self._self_similarity_score(device)
		# self.measure_img_rwd = self._image_reward_score(device)
		self.measure_clip_scr = self._clip_score(device)

	def evaluate_results(self, log_name: str, image_ref_prompt_pairs: List[Tuple[str, str, str]], self_simil: bool = True, img_reward: bool = True, clip: bool = True):
		self_simil_score = 0.0 if self_simil is True else None
		img_reward_score =  0.0 if img_reward is True else None
		clip_score =  0.0 if clip is True else None

		with open("images.txt", "w") as f:
			for item in image_ref_prompt_pairs:
				f.write(str(item) + "\n")


		for img_fp, ref_fp, prompt in tqdm(image_ref_prompt_pairs, desc="Evaluating images"):
			if self_simil_score is not None:
				self_simil_score += self.measure_self_sim(img_fp, ref_fp)
			if img_reward_score is not None:
				img_reward_score += self.measure_img_rwd(img_fp, prompt)
			if clip_score is not None:
				clip_score += self.measure_clip_scr(img_fp, prompt)

		if self_simil_score is not None:
			self_simil_score /= len(image_ref_prompt_pairs)
		if img_reward_score is not None:
			img_reward_score /= len(image_ref_prompt_pairs)
		if clip_score is not None:
			clip_score /= len(image_ref_prompt_pairs)

		print(f"######### {log_name} ########")
		if self_simil_score is not None:
			print(f"self similarity score: {self_simil_score}")
		if img_reward_score is not None:
			print(f"image reward score: {img_reward_score}")
		if clip_score is not None:
			print(f"clip score: {clip_score}")

		self._record_as_csv(log_name, self_simil_score, img_reward_score, clip_score)
