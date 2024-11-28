import os

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from ..utils import assert_path


def save_attention_maps(attn_maps, tokenizer, prompts, base_dir='log/attn_maps', unconditional=True, prefix=""):
	to_pil = ToPILImage()

	token_ids = tokenizer(prompts)['input_ids']
	total_tokens = []
	for token_id in token_ids:
		total_tokens.append(tokenizer.convert_ids_to_tokens(token_id))

	assert_path(base_dir)

	total_attn_map = list(list(attn_maps.values())[0].values())[0].sum(1)
	if unconditional:
		total_attn_map = total_attn_map.chunk(2)[1]  # (batch, height, width, attn_dim)
	total_attn_map = total_attn_map.permute(0, 3, 1, 2)
	total_attn_map = torch.zeros_like(total_attn_map)
	total_attn_map_shape = total_attn_map.shape[-2:]
	total_attn_map_number = 0

	for timestep, layers in attn_maps.items():
		timestep_dir = os.path.join(base_dir, f'{timestep}')
		assert_path(timestep_dir)

		for layer, attn_map in layers.items():
			layer_dir = os.path.join(timestep_dir, f'{layer}')
			assert_path(layer_dir)

			attn_map = attn_map.sum(1).squeeze(1)
			attn_map = attn_map.permute(0, 3, 1, 2)

			if unconditional:
				attn_map = attn_map.chunk(2)[1]

			resized_attn_map = F.interpolate(attn_map, size=total_attn_map_shape, mode='bilinear', align_corners=False)
			total_attn_map += resized_attn_map
			total_attn_map_number += 1

			for batch, (tokens, attn) in enumerate(zip(total_tokens, attn_map)):
				batch_dir = os.path.join(layer_dir, f'batch-{batch}')
				if not os.path.exists(batch_dir):
					os.mkdir(batch_dir)

				startofword = True
				for i, (token, a) in enumerate(zip(tokens, attn[:len(tokens)])):
					if '</w>' in token:
						token = token.replace('</w>', '')
						if startofword:
							token = '<' + token + '>'
						else:
							token = '-' + token + '>'
							startofword = True

					elif token != '<|startoftext|>' and token != '<|endoftext|>':
						if startofword:
							token = '<' + token + '-'
							startofword = False
						else:
							token = '-' + token + '-'

					to_pil(a.to(torch.float32)).save(os.path.join(batch_dir, f'{prefix}{i}-{token}.png'))

	total_attn_map /= total_attn_map_number
	for batch, (attn_map, tokens) in enumerate(zip(total_attn_map, total_tokens)):
		batch_dir = os.path.join(base_dir, f'batch-{batch}')
		if not os.path.exists(batch_dir):
			os.mkdir(batch_dir)

		startofword = True
		for i, (token, a) in enumerate(zip(tokens, attn_map[:len(tokens)])):
			if '</w>' in token:
				token = token.replace('</w>', '')
				if startofword:
					token = '<' + token + '>'
				else:
					token = '-' + token + '>'
					startofword = True

			elif token != '<|startoftext|>' and token != '<|endoftext|>':
				if startofword:
					token = '<' + token + '-'
					startofword = False
				else:
					token = '-' + token + '-'

			to_pil(a.to(torch.float32)).save(os.path.join(batch_dir, f'{i}-{token}.png'))
