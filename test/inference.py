import os
from typing import List, TypedDict

import torch
from controlnet_aux import CannyDetector, OpenposeDetector, ZoeDetector
from diffusers import AutoencoderKL, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
from pytorch_lightning import seed_everything

from lib import (assert_path, image_grid, init_store_attn_map,
                 save_alpha_masks, save_attention_maps)
from smartcontrol import SmartControlPipeline, register_unet
from .types import ModelType

class GenerateParam(TypedDict):
	seed: int
	prompt: str
	ignore_special_tkns: bool

class EvalModel():
	base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
	vae_model_path = "stabilityai/sd-vae-ft-mse"
	control: str = "depth"	# "depth" | "pose" | "canny"
	generate_param: GenerateParam = {}

	def _control_setup(self, control: str):
		if control == "depth":
			self.controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
			self.preprocessor = ZoeDetector.from_pretrained("lllyasviel/Annotators")
		elif control == "canny":
			self.controlnet_path = "lllyasviel/control_v11p_sd15_canny"
			self.preprocessor = CannyDetector()
		elif control == "pose":
			self.controlnet_path = "lllyasviel/control_v11p_sd15_openpose"
			self.preprocessor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

	def __init__(self, control: str):
		assert control in ["depth", "pose", "canny"], f"control argument should be one of depth, pose, canny, but you provied {control}"
		self.control = control

		self._control_setup(control)
		vae = AutoencoderKL.from_pretrained(self.vae_model_path).to(dtype=torch.float16)
		controlnet = ControlNetModel.from_pretrained(self.controlnet_path, torch_dtype=torch.float16)

		pipe =	SmartControlPipeline.from_pretrained(
				pretrained_model_name_or_path=self.base_model_path,
				controlnet=controlnet,
				vae=vae,
	   			torch_dtype=torch.float16
		)
		pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
		pipe.enable_model_cpu_offload()

		self.pipe = pipe

	def _prepare_control(self, reference: str):
		image = Image.open(reference)
		control_img = self.preprocessor(image)

		self.reference = image
		self.control_img = control_img
		return control_img

	def _SmartControl_decide_ckpt(self):
		if self.control == "depth":
			return f"{os.getcwd()}/depth.ckpt"
		elif self.control == "canny":
			return f"{os.getcwd()}/canny.ckpt"
		elif self.control == "pose":
			return None

	def _record_generate_params(self, seed: int,  prompt: str, ignore_special_tkns: bool):
		self.generate_param["seed"] = seed
		self.generate_param["prompt"] = prompt
		self.generate_param["ignore_special_tkns"] = ignore_special_tkns

	def get_inference_func(self, modelType: ModelType):
		self.modelType = modelType

		if modelType == ModelType.ControlNet:
			return self.inference_ControlNet
		elif modelType == ModelType.SmartControl:
			return self.inference_SmartControl
		elif modelType == ModelType.ControlAttend:
			return self.inference_ControlAttend

		return None

	def set_output_dir(self, output_dir: str):
		assert_path(output_dir)
		self.output_dir = output_dir

	@staticmethod
	def _filepath2name(filepath: str):
		no_extension = filepath.split(".")[0]
		name = " ".join(no_extension.split("/")[-2:])
		return name

	def _is_already_generated(self, output_name, seed):
		save_dir = f"{self.output_dir}/{self.modelType.name}/{output_name}"
		target = f"{save_dir}/generated - seed {seed}.png"

		return os.path.exists(target)

	def inference_ControlNet(self, prompt: str, reference: str, seed: int, alpha_mask: List[float] = [1.0], **kwargs):
		output_name = f"ControlNet {alpha_mask} - {prompt} with {self._filepath2name(reference)}"
		if self._is_already_generated(output_name, seed):
			print(f"{output_name} - seed {seed}  is already generated. Skipping.")
			return None, None

		control_img = self._prepare_control(reference)

		init_store_attn_map(self.pipe)
		register_unet(self.pipe, None, mask_options={
			"alpha_mask": alpha_mask,
			"fixed": True
		})

		self._record_generate_params(seed=seed, prompt=prompt, ignore_special_tkns=False)

		seed_everything(seed)
		output = self.pipe(
			prompt=prompt,
			image=control_img
		).images[0]

		return output, output_name

	def inference_SmartControl(self, prompt: str, reference: str, seed: int, **kwargs):
		output_name = f"SmartControl - {prompt} with {self._filepath2name(reference)}"
		if self._is_already_generated(output_name, seed):
			print(f"{output_name} - seed {seed} is already generated. Skipping.")
			return None, None

		control_img = self._prepare_control(reference)

		init_store_attn_map(self.pipe)
		register_unet(self.pipe,
				smart_ckpt= self._SmartControl_decide_ckpt(),
				mask_options={
					"alpha_mask": [1.0],
					"fixed": False
		})

		self._record_generate_params(seed=seed, prompt=prompt, ignore_special_tkns=False)

		seed_everything(seed)
		output = self.pipe(
			prompt=prompt,
			image=control_img
		).images[0]

		return output, output_name

	def inference_ControlAttend(self, prompt: str, reference: str, seed: int, mask_prompt, focus_tokens, **kwargs):
		output_name = f"ControlAttend - {prompt} with {self._filepath2name(reference)} focusing on {focus_tokens} of {mask_prompt}"
		if self._is_already_generated(output_name, seed):
			print(f"{output_name} - seed {seed}  is already generated. Skipping.")
			return None, None

		control_img = self._prepare_control(reference)

		pipe_options = {
			"ignore_special_tkns": True
		}
		self.pipe.options = pipe_options

		init_store_attn_map(self.pipe)
		register_unet(
			pipe=self.pipe,
			smart_ckpt=None,
			mask_options={
				"alpha_mask": [1.0],
				"fixed": True
			}
		)

		seed_everything(seed)

		self.pipe(
			prompt=mask_prompt,
			image=control_img,
			prepare_phase=True,
		)
		save_attention_maps(
			self.pipe.unet.attn_maps,
			self.pipe.tokenizer,
			base_dir=f"{os.getcwd()}/log/attn_maps/{self.modelType.name}//{output_name}/{mask_prompt}",
			prompts=[mask_prompt],
			options={
				"prefix": "",
				"return_dict": False,
				"ignore_special_tkns": True,
				"enabled_editing_prompts": 0
		})

		self._record_generate_params(seed=seed, prompt=prompt, ignore_special_tkns=True)

		register_unet(
			pipe=self.pipe,
			smart_ckpt=None,
			mask_options={
				"alpha_mask": [1.0],
				"fixed": True
			},
			reset_masks=False
		)
		output = self.pipe(
			prompt=prompt,
			mask_prompt=mask_prompt,
			focus_prompt = focus_tokens,
			image=control_img,
			prepare_phase=False
		).images[0]

		return output, output_name

	def postprocess(self, image, image_name, save_attn: bool = False):
		if image is None or image_name is None:
			return

		save_dir = f"{self.output_dir}/{self.modelType.name}/{image_name}"
		assert_path(save_dir)
		image.save(f"{save_dir}/generated - seed {self.generate_param['seed']}.png")
		comparison = image_grid([
		  		self.reference.resize((512, 512)),
				self.control_img.resize((512, 512)),
		 		image.resize((512, 512))
		   ], 1, 3)
		comparison.save(f"{save_dir}/control_result - seed {self.generate_param['seed']}.png")

		if save_attn:
			save_attention_maps(
			self.pipe.unet.attn_maps,
			self.pipe.tokenizer,
			base_dir=f"{os.getcwd()}/log/attn_maps/{self.modelType.name}/{image_name}",
			prompts=[self.generate_param["prompt"]],
			options={
				"prefix": "",
				"return_dict": False,
				"ignore_special_tkns": self.generate_param["ignore_special_tkns"],
				"enabled_editing_prompts": 0
			})
			save_alpha_masks(self.pipe.unet.alpha_masks, f'{os.getcwd()}/log/alpha_masks/{self.modelType.name}/{image_name}')

		print(f"Saved results for {image_name}")
