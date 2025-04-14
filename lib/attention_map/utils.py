import copy
from typing import Dict, List
from enum import Enum

import numpy as np
import torch

class AggMode(Enum):
	Mean = 1
	Norm = 2



def _aggregate(attns: Dict[str, torch.Tensor], focus_indexes: List[int]) -> Dict[str, torch.Tensor]:
	agg = dict()

	for block, v in attns.items():
		multihead_avg = v.sum(1) / v.shape[1] # batch attn_head h w attn_dim -> batch h w attn_dim
		multihead_avg = multihead_avg.permute(0, 3, 1, 2)	# batch h w attn_dim -> batch attn_dim h w
		multihead_avg = multihead_avg.chunk(2)[1]
		focused = multihead_avg[:, focus_indexes, :, :]

		block_name = block.split(".")[1]

		try:
			block_num = int(block.split(".")[2])
		# if there is no block number (single block case)
		except ValueError:
			block_num = 0

		if block_name not in agg:
			agg[block_name] = dict()

		if block_num not in agg[block_name]:
			agg[block_name][block_num] = focused
		else:
			# a block can have multiple attention modules
			agg[block_name][block_num] = torch.cat((agg[block_name][block_num], focused))

	return agg

def _average(agg_res: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
	agg = copy.deepcopy(agg_res)

	for block, CAs in agg_res.items():
		for block_num, attn in CAs.items():
			assert isinstance(attn, torch.Tensor), "aggregation should return a dictionary who has a tensor as value"
			# mean multiple focus token's attention values
			attn = attn.mean(dim=1)

			# mean multiple attention modules' results
			agg[block][block_num] = attn.mean(dim=0)
			# agg[block][block_num][agg[block][block_num] < 0.2] = 0
			# agg[block][block_num][agg[block][block_num] > 0.2] = 0.4

	return agg

def _normalize(agg_res: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
	agg = copy.deepcopy(agg_res)

	for block, CAs in agg_res.items():
		for block_num, attn in CAs.items():
			assert isinstance(attn, torch.Tensor), "aggregation should return a dictionary who has a tensor as value"

			# mean multiple attention modules' results
			attn = attn.mean(dim=0)
			# cutoff each token's attention values
			for i in range(attn.shape[0]):
				h, w = attn[i].shape
				flatten = attn[i].flatten()
				_, tail_idx = torch.topk(flatten, int(flatten.numel() * 0.5), largest=False)
				flatten[tail_idx] = 0
				flatten /= flatten.max()
				attn[i] =flatten.reshape((h, w))

			# max pool on prompt dimension
			aggregated , _ = torch.max(attn, dim=0)
			agg[block][block_num] = aggregated

	return agg

def _get_aggregation_function(agg_mode: AggMode):
	if agg_mode == AggMode.Mean:
		return _average
	elif agg_mode == AggMode.Norm:
		return _normalize

def agg_by_blocks(attns: Dict[str, torch.Tensor], focus_indexes: List[int], agg_mode: AggMode = AggMode.Mean):
	agg = _aggregate(attns, focus_indexes)
	post_process = _get_aggregation_function(agg_mode)
	aggregated = post_process(agg)

	return aggregated

def agg_mode_str2enum(agg_mode: str):
	if agg_mode == "norm":
		return AggMode.Norm
	return AggMode.Mean

def stringfy_agg_mode(agg_mode: AggMode):
    if agg_mode == AggMode.Mean:
        return ""
    elif agg_mode == AggMode.Norm:
        return "normalized"
