
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.transformer_2d import Transformer2DModel

from .mocks import (BasicTransformerBlockForward, Transformer2DModelForward,
                    attn_call)


def hook_function(name, detach=True):
	def forward_hook(module, input, output):
		assert hasattr(module, "attn_maps"), f"Attn map implanted module {module.__class__.__name__} should have a class member `attn_maps` to store attention maps"

		if hasattr(module.processor, "attn_map"):
			timestep = module.processor.timestep

			module.attn_maps[timestep] = module.attn_maps.get(timestep, dict())
			module.attn_maps[timestep][name] = module.processor.attn_map.cpu() if detach \
				else module.processor.attn_map

			del module.processor.attn_map

	return forward_hook

def register_cross_attention_hook(model, target_name):
	for name, module in model.named_modules():
		if not name.endswith(target_name):
			continue

		# create a new flag instance variable to indicate that attention map should be stored
		if isinstance(module.processor, AttnProcessor):
			module.processor.store_attn_map = True # type: ignore
		"""
		elif isinstance(module.processor, AttnProcessor2_0):
			module.processor.store_attn_map = True
		elif isinstance(module.processor, LoRAAttnProcessor):
			module.processor.store_attn_map = True
		elif isinstance(module.processor, LoRAAttnProcessor2_0):
			module.processor.store_attn_map = True
		elif isinstance(module.processor, JointAttnProcessor2_0):
			module.processor.store_attn_map = True
		elif isinstance(module.processor, FluxAttnProcessor2_0):
			module.processor.store_attn_map = True
		"""

		hook = module.register_forward_hook(hook_function(name))

	return model

def replace_call_method_for_unet(model):
	#if model.__class__.__name__ == 'UNet2DConditionModel':
		#model.forward = UNet2DConditionModelForward.__get__(model, UNet2DConditionModel)

	for _, layer in model.named_children():
		if layer.__class__.__name__ == 'Transformer2DModel':
			layer.forward = Transformer2DModelForward.__get__(layer, Transformer2DModel)

		elif layer.__class__.__name__ == 'BasicTransformerBlock':
			layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)

		replace_call_method_for_unet(layer)

	return model

def init_cross_attn():
	AttnProcessor.__call__ = attn_call

def init_pipeline(pipeline):
	"""
	if 'transformer' in vars(pipeline).keys():
		if pipeline.transformer.__class__.__name__ == 'SD3Transformer2DModel':
			pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
			pipeline.transformer = replace_call_method_for_sd3(pipeline.transformer)

		elif pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
			FluxPipeline.__call__ = FluxPipeline_call
			pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
			pipeline.transformer = replace_call_method_for_flux(pipeline.transformer)
	else:
	"""

	if pipeline.unet.__class__.__name__ == 'UNet2DConditionModel':
		# save attention maps in this class member
		pipeline.unet.attn_maps = {}
		# attn2 processor takes charge of the cross-attention
		pipeline.unet = register_cross_attention_hook(pipeline.unet, 'attn2')
		pipeline.unet = replace_call_method_for_unet(pipeline.unet)


	return pipeline

def init_store_attn_map(pipeline):
	init_cross_attn()
	return init_pipeline(pipeline=pipeline)
